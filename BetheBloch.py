from packages import *
from scipy.interpolate import CubicSpline

class BetheBloch:

    def __init__(self, pdg=0):
        self.pdgcode = pdg
        self.mass = 0
        self.charge = 0
        self.sp_KE_range = None
        self.sp_range_KE = None
        if pdg != 0:
            self.set_pdg_code(pdg)

    def set_pdg_code(self, pdg):
        self.pdgcode = pdg

        if abs(self.pdgcode) == 13:  # muon
            self.mass = 105.6583755
            self.charge = 1
        elif abs(self.pdgcode) == 211:  # pion
            self.mass = 139.57039
            self.charge = 1
        elif abs(self.pdgcode) == 321:  # kaon
            self.mass = 493.677
            self.charge = 1
        elif self.pdgcode == 2212:  # proton
            self.mass = 938.27208816
            self.charge = 1
        else:
            print(f"Unknown pdg code {self.pdgcode}")
            exit(1)

        self.create_splines()

    def density_effect(self, beta, gamma):
        lar_C = 5.215
        lar_x0 = 0.201
        lar_x1 = 3
        lar_a = 0.196
        lar_k = 3
        x = math.log10(beta * gamma)

        if x >= lar_x1:
            return 2 * math.log(10) * x - lar_C
        elif lar_x0 <= x < lar_x1:
            return 2 * math.log(10) * x - lar_C + lar_a * (lar_x1 - x) ** lar_k
        else:
            return 0  # if x < lar_x0

    def beta_gamma(self, KE):
        gamma = (KE + self.mass) / self.mass
        beta = math.sqrt(1 - 1 / gamma ** 2)
        return beta * gamma

    def mean_dEdx(self, KE):
        K = 0.307
        rho = 1.396
        Z = 18
        A = 39.948
        I = 10.5 * 18 * 1e-6  # MeV
        me = 0.511  # MeV

        gamma = (KE + self.mass) / self.mass
        beta = math.sqrt(1 - 1 / gamma ** 2)
        wmax = 2 * me * beta ** 2 * gamma ** 2 / (1 + 2 * gamma * me / self.mass + (me / self.mass) ** 2)

        dEdX = (rho * K * Z * self.charge ** 2) / (A * beta ** 2) * (
                0.5 * math.log(2 * me * gamma ** 2 * beta ** 2 * wmax / I ** 2) - beta ** 2 - self.density_effect(beta, gamma) / 2)
        return dEdX

    def mpv_dEdx(self, KE, pitch):
        K = 0.307
        rho = 1.396
        Z = 18
        A = 39.948
        I = 10.5 * 18 * 1e-6  # MeV
        me = 0.511  # MeV

        gamma = (KE + self.mass) / self.mass
        beta = math.sqrt(1 - 1 / gamma ** 2)
        xi = (K / 2) * (Z / A) * (pitch * rho / beta ** 2)
        eloss_mpv = xi * (math.log(2 * me * gamma ** 2 * beta ** 2 / I) + math.log(xi / I) + 0.2 - beta ** 2 - self.density_effect(beta, gamma)) / pitch

        return eloss_mpv

    def integrated_dEdx(self, KE0, KE1, num_points=10000):
        if KE0 > KE1:
            KE0, KE1 = KE1, KE0

        step = (KE1 - KE0) / num_points
        area = 0

        for i in range(num_points):
            dEdx = self.mean_dEdx(KE0 + (i + 0.5) * step)
            if dEdx:
                area += 1 / dEdx * step
        return area

    def range_from_KE(self, KE):
        return self.integrated_dEdx(0, KE)

    def create_splines(self, num_points=1000, minke=0.1, maxke=1e4):
        if self.sp_KE_range is not None:
            self.sp_KE_range = None
        if self.sp_range_KE is not None:
            self.sp_range_KE = None

        KE = np.logspace(math.log10(minke), math.log10(maxke), num_points)
        Range = np.array([self.range_from_KE(ke) for ke in KE])

        # special treatment added to solve non-increasing sequence in CubicSpline due to precision (C++ spline function seems to ignore this problem)
        indices_to_delete = np.arange(1, num_points)[np.diff(Range)<0]
        while len(indices_to_delete):
            KE = np.delete(KE, indices_to_delete)
            Range = np.delete(Range, indices_to_delete)
            indices_to_delete = np.arange(1, len(Range))[np.diff(Range)<0]

        self.sp_KE_range = CubicSpline(KE, Range)
        self.sp_range_KE = CubicSpline(Range, KE)
        print(f"Done creating splines for particle with pdgcode {self.pdgcode}")

    def range_from_KE_spline(self, KE):
        if self.sp_KE_range is None:
            print("Spline does not exist.")
            exit(1)
        return self.sp_KE_range(KE)

    def KE_from_range_spline(self, range):
        if self.sp_range_KE is None:
            print("Spline does not exist.")
            exit(1)
        return self.sp_range_KE(range)

    def KE_at_length(self, KE0, tracklength):
        trklen0 = self.range_from_KE_spline(KE0)
        trklen1 = trklen0 - tracklength
        if trklen1 > 0:
            return self.KE_from_range_spline(trklen1)
        else:
            return 0


if __name__ == "__main__":
    bb = BetheBloch(211)
    print(bb.mean_dEdx(100))
    print(bb.range_from_KE_spline(100), bb.range_from_KE_spline(100.3), bb.range_from_KE_spline(235.6), bb.range_from_KE_spline(789.6), bb.KE_from_range_spline(100.5), bb.KE_from_range_spline(40.3), bb.KE_from_range_spline(10.3), bb.KE_from_range_spline(2))
    print(bb.KE_at_length(400.7,26.7))
