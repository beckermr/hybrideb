import numpy as np
import scipy.special
import scipy.interpolate
import copy
import tqdm

from ._dblquad import dblquad

try:
    import pygsl.sf

    HAVE_GSL = True
except ImportError:
    import scipy.special

    HAVE_GSL = False


class HybridEB(object):
    def __init__(
        self, tmin_in, tmax_in, Nbins, linear=False, useArcmin=True, windows=None
    ):
        # numerics
        self.epsabs = 1e-1
        self.epsrel = 1e-1

        # get bin info
        if useArcmin:
            am2rad = np.pi / 180.0 / 60.0
        else:
            am2rad = 1.0
        self.L = tmin_in * am2rad
        self.H = tmax_in * am2rad
        self.Nb = Nbins

        if linear:
            self.Lb = np.arange(self.Nb) * (self.H - self.L) / self.Nb + self.L
            self.Hb = (np.arange(self.Nb) + 1.0) * (self.H - self.L) / self.Nb + self.L
        else:
            self.Lb = (
                np.exp(np.arange(self.Nb) * np.log(self.H / self.L) / self.Nb) * self.L
            )
            self.Hb = (
                np.exp((np.arange(self.Nb) + 1.0) * np.log(self.H / self.L) / self.Nb)
                * self.L
            )

        # make the bin window functions
        if windows is None:

            def _make_geomwin(L, H):
                return lambda x: 2.0 * x / (H * H - L * L)

            self.windows = []
            for i in range(self.Nb):
                self.windows.append(_make_geomwin(self.Lb[i], self.Hb[i]))
        else:

            def _make_normwin(winf, norm):
                return lambda x: winf(x / am2rad) / norm

            self.windows = []
            assert (
                len(windows) == self.Nb
            ), "hybridEB requires as many windows as angular bins!"

            for i in range(self.Nb):
                twin = _make_normwin(windows[i], 1.0)
                norm, err = scipy.integrate.quad(twin, self.Lb[i], self.Hb[i])
                self.windows.append(_make_normwin(windows[i], norm))

    def j0_int(self, t, args):
        lv = args[0]
        win = args[1]
        ellwin = args[2]
        if HAVE_GSL:
            return (
                (pygsl.sf.bessel_J0(lv * t))[0] * win(t) * ellwin(lv) * lv / 2.0 / np.pi
            )
        else:
            return (scipy.special.j0(lv * t)) * win(t) * ellwin(lv) * lv / 2.0 / np.pi

    def j4_int(self, t, args):
        lv = args[0]
        win = args[1]
        ellwin = args[2]

        if HAVE_GSL:
            return (
                (pygsl.sf.bessel_Jn(4, lv * t))[0]
                * win(t)
                * ellwin(lv)
                * lv
                / 2.0
                / np.pi
            )
        else:
            return (
                (scipy.special.jn(4, lv * t)) * win(t) * ellwin(lv) * lv / 2.0 / np.pi
            )

    def ellwin0(
        self,
        ellwin,
        bini,
        ellrange=(1.0, 1e5),
        epsrel=1e-6,
        epsabs=1e-6,
    ):
        args = [self.windows[bini], ellwin]
        return dblquad(
            self.j0_int,
            ellrange[0],
            ellrange[1],
            lambda x: self.Lb[bini],
            lambda x: self.Hb[bini],
            epsabs=self.epsabs,
            epsrel=self.epsrel,
            args=args,
        )

    def ellwin4(
        self,
        ellwin,
        bini,
        ellrange=(1.0, 1e5),
        epsrel=1e-6,
        epsabs=1e-6,
    ):
        args = [self.windows[bini], ellwin]
        return dblquad(
            self.j4_int,
            ellrange[0],
            ellrange[1],
            lambda x: self.Lb[bini],
            lambda x: self.Hb[bini],
            epsabs=self.epsabs,
            epsrel=self.epsrel,
            args=args,
        )

    def make_est(
        self,
        ellwin,
        beb,
        sepEB=False,
        ellrange=(1.0, 1e5),
        Naint=10,
    ):
        A0 = np.zeros(beb.Nb)
        A4 = np.zeros(beb.Nb)

        iranges = np.linspace(ellrange[0], ellrange[1], int(Naint + 1))
        for i in range(beb.Nb):
            A0[i] = 0.0
            A4[i] = 0.0
            for j in range(int(Naint)):
                A0[i] += self.ellwin0(ellwin, i, ellrange=(iranges[j], iranges[j + 1]))
                A4[i] += self.ellwin4(ellwin, i, ellrange=(iranges[j], iranges[j + 1]))
            A0[i] = 2.0 * A0[i] / beb.invnorm[i] * 2.0 * np.pi
            A4[i] = 2.0 * A4[i] / beb.invnorm[i] * 2.0 * np.pi

        if sepEB:
            mat = np.identity(beb.Nb) + np.dot(
                np.dot(beb.mm, beb.mp),
                np.identity(beb.Nb)
                - np.outer(beb.fa_on, beb.fa_on)
                - np.outer(beb.fb_on, beb.fb_on),
            )
            invmat = np.linalg.inv(mat)
            fptest = np.dot(invmat, A0)
            fp, fm = beb.fplusminus(fptest)
        else:
            mat = np.identity(beb.Nb) - np.dot(beb.mm, beb.mp)
            invmat = np.linalg.inv(mat)
            fp = np.dot(invmat, A0 - np.dot(beb.mm, A4))
            fm = A4 - np.dot(beb.mp, fp)
            fptest = fp.copy()

        ell, wp, wm = beb.wplusminus(fp, fm)

        return np.sqrt(beb.Lb * beb.Hb), fp, fm, ell, wp, wm


class GaussEB(object):
    def __init__(
        self, bEB, hEB, Nl=20, lmin=100.0, lmax=3000.0, sepEB=True, fname=None
    ):
        if fname is not None:
            self.read_data(fname)
        else:
            # make sure objects agree
            assert bEB.L == hEB.L
            assert bEB.H == hEB.H
            assert bEB.Nb == hEB.Nb

            self.Nb = copy.copy(bEB.Nb)
            self.L = copy.copy(bEB.L)
            self.H = copy.copy(bEB.H)
            self.Lb = bEB.Lb.copy()
            self.Hb = bEB.Hb.copy()

            def make_win(lm, ls):
                return lambda _l: self.gausswin(_l, lm, ls)

            self.Nl = Nl
            self.lmin = lmin
            self.lmax = lmax
            self.lbmin = (
                np.arange(self.Nl) * (self.lmax - self.lmin) / self.Nl + self.lmin
            )
            self.lbmax = (np.arange(self.Nl) + 1.0) * (
                self.lmax - self.lmin
            ) / self.Nl + self.lmin
            self.lm = (self.lbmin + self.lbmax) / 2.0
            self.ls = np.log(self.lbmax / self.lbmin) / 2.0

            self._Nint = []
            self._ellL = []
            self._ellH = []
            self._ellwin = []
            fac = 10.0
            for i in tqdm.trange(self.Nl, desc="making GaussEB ell windows", ncols=80):
                self._Nint.append(3 + i / 4)
                self._ellwin.append(make_win(self.lm[i], self.ls[i]))
                if self.lm[i] - fac * self.ls[i] * self.lm[i] <= 0.0:
                    self._ellL.append(1.0)
                else:
                    self._ellL.append(self.lm[i] - fac * self.ls[i] * self.lm[i])
                self._ellH.append(self.lm[i] + fac * self.ls[i] * self.lm[i])

            self.fp = []
            self.fm = []
            self.wp = []
            self.wm = []
            loc = 0
            for ellwin, ellL, ellH, Nint in tqdm.tqdm(
                zip(self._ellwin, self._ellL, self._ellH, self._Nint),
                desc="making GaussEB estimators",
                ncols=80,
                total=len(self._ellwin),
            ):
                r, fp, fm, ell, wp, wm = hEB.make_est(
                    ellwin,
                    bEB,
                    sepEB=sepEB,
                    ellrange=(ellL, ellH),
                    Naint=Nint,
                )
                self.r = r
                self.ell = ell
                self.fp.append(fp)
                self.fm.append(fm)
                self.wp.append(wp)
                self.wm.append(wm)
                loc += 1

    def gausswin(self, lv, lm, ls):
        xdev = np.log(lv / lm) / ls
        return np.exp(-0.5 * xdev * xdev) / np.sqrt(2.0 * np.pi) / ls

    def __call__(self, i):
        assert i >= 0 and i < self.Nl
        return self.r, self.fp[i], self.fm[i], self.ell, self.wp[i], self.wm[i]

    def _io_vec(self, fp, vec, name, read=False):
        if read:
            line = fp.readline()
            line = fp.readline()
            line = line.strip()
            val = np.array([float(tag) for tag in line.split()])
            line = fp.readline()
            return val
        else:
            fp.write("# %s\n" % name)
            for val in vec:
                fp.write("%.20lg " % val)
            fp.write("\n#\n")
            return vec

    def _io_data(self, fname, read=False):
        if read:
            self.lbmin = None
            self.lbmax = None
            self.lm = None
            self.ls = None
            self.Lb = None
            self.Hb = None
            self.r = None
            self.ell = None

            fp = open(fname, "r")
            line = fp.readline()
            line = fp.readline()
            line = line.strip()
            line = line.split()
            self.Nl = int(line[0])
            self.lmin = float(line[1])
            self.lmax = float(line[2])

            self.fp = []
            self.fm = []
            self.wp = []
            self.wm = []
            for i in range(self.Nl):
                self.fp.append(None)
                self.fm.append(None)
                self.wp.append(None)
                self.wm.append(None)

            line = fp.readline()
            line = fp.readline()
            line = line.strip()
            line = line.split()
            self.Nb = int(line[0])
            self.L = float(line[1])
            self.H = float(line[2])
        else:
            fp = open(fname, "w")
            fp.write("# Nl lmin lmax\n")
            fp.write("%ld %.20lg %.20lg\n" % (self.Nl, self.lmin, self.lmax))
            fp.write("# N L H\n")
            fp.write("%ld %.20lg %.20lg\n" % (self.Nb, self.L, self.H))

        self.lbmin = self._io_vec(fp, self.lbmin, "lbmin", read=read)
        self.lbmax = self._io_vec(fp, self.lbmax, "lbmax", read=read)
        self.lm = self._io_vec(fp, self.lm, "lm", read=read)
        self.ls = self._io_vec(fp, self.ls, "ls", read=read)

        self.Lb = self._io_vec(fp, self.Lb, "Lb", read=read)
        self.Hb = self._io_vec(fp, self.Hb, "Hb", read=read)

        self.r = self._io_vec(fp, self.r, "r", read=read)
        self.ell = self._io_vec(fp, self.ell, "ell", read=read)

        for i in range(self.Nl):
            self.fp[i] = self._io_vec(fp, self.fp[i], "fp%d" % i, read=read)
        for i in range(self.Nl):
            self.fm[i] = self._io_vec(fp, self.fm[i], "fm%d" % i, read=read)

        for i in range(self.Nl):
            self.wp[i] = self._io_vec(fp, self.wp[i], "wp%d" % i, read=read)
        for i in range(self.Nl):
            self.wm[i] = self._io_vec(fp, self.wm[i], "wm%d" % i, read=read)

        fp.close()

    def write_data(self, fname):
        self._io_data(fname, read=False)

    def read_data(self, fname):
        self._io_data(fname, read=True)


class SimpleGaussEB(object):
    def __init__(self, bEB, Nl=20, lmin=100.0, lmax=3000.0, fname=None):
        if fname is not None:
            self.read_data(fname)
        else:
            self.r = (
                np.exp((np.arange(bEB.Nb) + 0.5) * np.log(bEB.H / bEB.L) / bEB.Nb)
                * bEB.L
            )

            self.Nl = Nl
            self.lmin = lmin
            self.lmax = lmax
            self.lbmin = (
                np.arange(self.Nl) * (self.lmax - self.lmin) / self.Nl + self.lmin
            )
            self.lbmax = (np.arange(self.Nl) + 1.0) * (
                self.lmax - self.lmin
            ) / self.Nl + self.lmin
            self.lm = (self.lbmin + self.lbmax) / 2.0
            self.ls = np.log(self.lbmax / self.lbmin) / 2.0

            self.Nb = copy.copy(bEB.Nb)
            self.L = copy.copy(bEB.L)
            self.H = copy.copy(bEB.H)
            self.Lb = bEB.Lb.copy()
            self.Hb = bEB.Hb.copy()

            self.fp = []
            self.fm = []
            self.wp = []
            self.wm = []
            for i in tqdm.trange(
                self.Nl, desc="making SimpleGaussEB ell windows", ncols=80
            ):
                lm = self.lm[i]
                ls = self.ls[i]
                fpi = []

                for j in tqdm.trange(bEB.Nb):
                    y0, err = scipy.integrate.quad(
                        self.j0intfun_gauss, 0.0, 1e2, args=(self.r[j], lm, ls)
                    )
                    y1, err = scipy.integrate.quad(
                        self.j0intfun_gauss, 1e2, 1e4, args=(self.r[j], lm, ls)
                    )
                    y2, err = scipy.integrate.quad(
                        self.j0intfun_gauss, 1e4, 1e6, args=(self.r[j], lm, ls)
                    )
                    fpi.append((y0 + y1 + y2) * (self.r[j] / 2.0 / np.pi) ** 2.0)
                fpi = np.array(fpi)

                fp, fm = bEB.fplusminus(fpi)
                self.fp.append(fp)
                self.fm.append(fm)
                ell, wp, wm = bEB.wplusminus(fp, fm)
                self.wp.append(wp)
                self.wm.append(wm)
                self.ell = ell

    def j0intfun_gauss(self, lv, t, lm, ls):
        return (
            scipy.special.jn(0, t * lv)
            * lv
            / 2.0
            / np.pi
            * np.exp(-0.5 * ((np.log(lv / lm)) / ls) ** 2.0)
            / np.sqrt(2.0 * np.pi)
            / ls
        )

    def j4intfun_gauss(self, lv, t, lm, ls):
        return (
            scipy.special.jn(4, t * lv)
            * lv
            / 2.0
            / np.pi
            * np.exp(-0.5 * ((np.log(lv / lm)) / ls) ** 2.0)
            / np.sqrt(2.0 * np.pi)
            / ls
        )

    def __call__(self, i):
        assert i >= 0 and i < self.Nl
        return self.r, self.fp[i], self.fm[i], self.ell, self.wp[i], self.wm[i]

    def _io_vec(self, fp, vec, name, read=False):
        if read:
            line = fp.readline()
            line = fp.readline()
            line = line.strip()
            val = np.array([float(tag) for tag in line.split()])
            line = fp.readline()
            return val
        else:
            fp.write("# %s\n" % name)
            for val in vec:
                fp.write("%.20lg " % val)
            fp.write("\n#\n")
            return vec

    def _io_data(self, fname, read=False):
        if read:
            self.lbmin = None
            self.lbmax = None
            self.lm = None
            self.ls = None
            self.Lb = None
            self.Hb = None
            self.r = None
            self.ell = None

            fp = open(fname, "r")
            line = fp.readline()
            line = fp.readline()
            line = line.strip()
            line = line.split()
            self.Nl = int(line[0])
            self.lmin = float(line[1])
            self.lmax = float(line[2])

            self.fp = []
            self.fm = []
            self.wp = []
            self.wm = []
            for i in range(self.Nl):
                self.fp.append(None)
                self.fm.append(None)
                self.wp.append(None)
                self.wm.append(None)

            line = fp.readline()
            line = fp.readline()
            line = line.strip()
            line = line.split()
            self.Nb = int(line[0])
            self.L = float(line[1])
            self.H = float(line[2])
        else:
            fp = open(fname, "w")
            fp.write("# Nl lmin lmax\n")
            fp.write("%ld %.20lg %.20lg\n" % (self.Nl, self.lmin, self.lmax))
            fp.write("# N L H\n")
            fp.write("%ld %.20lg %.20lg\n" % (self.Nb, self.L, self.H))

        self.lbmin = self._io_vec(fp, self.lbmin, "lbmin", read=read)
        self.lbmax = self._io_vec(fp, self.lbmax, "lbmax", read=read)
        self.lm = self._io_vec(fp, self.lm, "lm", read=read)
        self.ls = self._io_vec(fp, self.ls, "ls", read=read)

        self.Lb = self._io_vec(fp, self.Lb, "Lb", read=read)
        self.Hb = self._io_vec(fp, self.Hb, "Hb", read=read)

        self.r = self._io_vec(fp, self.r, "r", read=read)
        self.ell = self._io_vec(fp, self.ell, "ell", read=read)

        for i in range(self.Nl):
            self.fp[i] = self._io_vec(fp, self.fp[i], "fp%d" % i, read=read)
        for i in range(self.Nl):
            self.fm[i] = self._io_vec(fp, self.fm[i], "fm%d" % i, read=read)

        for i in range(self.Nl):
            self.wp[i] = self._io_vec(fp, self.wp[i], "wp%d" % i, read=read)
        for i in range(self.Nl):
            self.wm[i] = self._io_vec(fp, self.wm[i], "wm%d" % i, read=read)

        fp.close()

    def write_data(self, fname):
        self._io_data(fname, read=False)

    def read_data(self, fname):
        self._io_data(fname, read=True)
