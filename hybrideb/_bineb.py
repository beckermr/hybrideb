import numpy as np
import scipy.integrate
import scipy.special
import tqdm

from ._dblquad import dblquad

HAVE_PYGSL = False
try:
    import pygsl.integrate
    import pygsl.sf

    HAVE_PYGSL = True
except ImportError:
    pass


class BinEB(object):
    def __init__(
        self, tmin, tmax, Nb, windows=None, linear=False, useArcmin=True, fname=None
    ):
        if fname is not None:
            self.read_data(fname)
        else:
            # set basic params
            if useArcmin:
                am2r = np.pi / 180.0 / 60.0
            else:
                am2r = 1.0
            self.Nb = Nb
            self.L = tmin * am2r
            self.H = tmax * am2r
            if linear:
                self.Lb = (self.H - self.L) / Nb * np.arange(Nb) + self.L
                self.Hb = (self.H - self.L) / Nb * (np.arange(Nb) + 1.0) + self.L
            else:
                self.Lb = np.exp(np.log(self.H / self.L) / Nb * np.arange(Nb)) * self.L
                self.Hb = (
                    np.exp(np.log(self.H / self.L) / Nb * (np.arange(Nb) + 1.0))
                    * self.L
                )
            self.have_ell_win = False

            # make the bin window functions
            if windows is None:

                def _make_geomwin(L, H):
                    return lambda x: 2.0 * x / (H * H - L * L)

                self.windows = []
                for i in range(self.Nb):
                    self.windows.append(_make_geomwin(self.Lb[i], self.Hb[i]))
            else:

                def _make_normwin(winf, norm):
                    return lambda x: winf(x / am2r) / norm

                self.windows = []
                assert (
                    len(windows) == Nb
                ), "binEB requires as many windows as angular bins!"

                for i in range(self.Nb):
                    twin = _make_normwin(windows[i], 1.0)
                    norm, err = scipy.integrate.quad(twin, self.Lb[i], self.Hb[i])
                    self.windows.append(_make_normwin(windows[i], norm))

            # get fa and fb
            self.fa = np.zeros(self.Nb)
            self.fa[:] = 1.0

            if HAVE_PYGSL:
                limit = 10
                epsabs = 1e-8
                epsrel = 1e-8
                w = pygsl.integrate.workspace(limit)

                def fb_int(x, args):
                    win = args[0]
                    return win(x) * x * x

                self.fb = np.zeros(self.Nb)
                for i in range(self.Nb):
                    args = [self.windows[i]]
                    f = pygsl.integrate.gsl_function(fb_int, args)
                    code, val, err = pygsl.integrate.qags(
                        f, self.Lb[i], self.Hb[i], epsabs, epsrel, limit, w
                    )
                    self.fb[i] = val
            else:

                def fb_int(x, win):
                    return win(x) * x * x

                self.fb = np.zeros(self.Nb)
                for i in range(self.Nb):
                    val, err = scipy.integrate.quad(
                        fb_int, self.Lb[i], self.Hb[i], args=(self.windows[i],)
                    )
                    self.fb[i] = val

            self.fa_on = self.fa / np.sqrt(np.sum(self.fa * self.fa))
            self.fb_on = self.fb - self.fa * np.sum(self.fa * self.fb) / np.sum(
                self.fa * self.fa
            )
            self.fb_on = self.fb_on / np.sqrt(np.sum(self.fb_on * self.fb_on))

            # get Mplus matrix
            if HAVE_PYGSL:
                limit = 10
                epsabs = 1e-8
                epsrel = 1e-8
                w = pygsl.integrate.workspace(limit)

                def knorm_int(x, args):
                    win = args[0]
                    return win(x) * win(x) / x

                knorm = np.zeros(self.Nb)
                for i in range(self.Nb):
                    args = [self.windows[i]]
                    f = pygsl.integrate.gsl_function(knorm_int, args)
                    code, val, err = pygsl.integrate.qags(
                        f, self.Lb[i], self.Hb[i], epsabs, epsrel, limit, w
                    )
                    knorm[i] = val
                self.invnorm = knorm

                def inv2_int(x, args):
                    win = args[0]
                    return win(x) / x / x

                inv2 = np.zeros(self.Nb)
                for i in range(self.Nb):
                    args = [self.windows[i]]
                    f = pygsl.integrate.gsl_function(inv2_int, args)
                    code, val, err = pygsl.integrate.qags(
                        f, self.Lb[i], self.Hb[i], epsabs, epsrel, limit, w
                    )
                    inv2[i] = val

                def inv4_int(x, args):
                    win = args[0]
                    return win(x) / x / x / x / x

                inv4 = np.zeros(self.Nb)
                for i in range(self.Nb):
                    args = [self.windows[i]]
                    f = pygsl.integrate.gsl_function(inv4_int, args)
                    code, val, err = pygsl.integrate.qags(
                        f, self.Lb[i], self.Hb[i], epsabs, epsrel, limit, w
                    )
                    inv4[i] = val
            else:

                def knorm_int(x, win):
                    return win(x) * win(x) / x

                knorm = np.zeros(self.Nb)
                for i in range(self.Nb):
                    val, err = scipy.integrate.quad(
                        knorm_int, self.Lb[i], self.Hb[i], args=(self.windows[i],)
                    )
                    knorm[i] = val
                self.invnorm = knorm

                def inv2_int(x, win):
                    return win(x) / x / x

                inv2 = np.zeros(self.Nb)
                for i in range(self.Nb):
                    val, err = scipy.integrate.quad(
                        inv2_int, self.Lb[i], self.Hb[i], args=(self.windows[i],)
                    )
                    inv2[i] = val

                def inv4_int(x, win):
                    return win(x) / x / x / x / x

                inv4 = np.zeros(self.Nb)
                for i in range(self.Nb):
                    val, err = scipy.integrate.quad(
                        inv4_int, self.Lb[i], self.Hb[i], args=(self.windows[i],)
                    )
                    inv4[i] = val

            if HAVE_PYGSL:

                def _mp_int(p, args):
                    t = args[0]
                    k = args[1]
                    i = args[2]
                    if p > t:
                        val = (
                            (4.0 / p / p - 12.0 * t * t / p / p / p / p)
                            * self.windows[k](p)
                            * self.windows[i](t)
                        )
                    else:
                        val = 0.0
                    return val

            else:

                def _mp_int(p, t, k, i):
                    if p > t:
                        return (
                            (4.0 / p / p - 12.0 * t * t / p / p / p / p)
                            * self.windows[k](p)
                            * self.windows[i](t)
                        )
                    else:
                        return 0.0

            self.mp = np.zeros((self.Nb, self.Nb))
            for k in range(self.Nb):
                for i in range(self.Nb):
                    if windows is None:
                        if i < k:
                            self.mp[k, i] += (
                                2.0
                                / (self.Hb[i] * self.Hb[i] - self.Lb[i] * self.Lb[i])
                                * (
                                    2.0
                                    * (
                                        self.Hb[i] * self.Hb[i]
                                        - self.Lb[i] * self.Lb[i]
                                    )
                                    * np.log(self.Hb[k] / self.Lb[k])
                                    + 3.0
                                    / 2.0
                                    * (
                                        np.power(self.Hb[i], 4.0)
                                        - np.power(self.Lb[i], 4.0)
                                    )
                                    * (
                                        1.0 / self.Hb[k] / self.Hb[k]
                                        - 1.0 / self.Lb[k] / self.Lb[k]
                                    )
                                )
                            )
                        if k == i:
                            self.mp[k, i] += 1.0
                            self.mp[k, i] += (
                                2.0
                                / (self.Hb[i] * self.Hb[i] - self.Lb[i] * self.Lb[i])
                                * (
                                    -0.5
                                    * (
                                        self.Hb[k] * self.Hb[k]
                                        - self.Lb[k] * self.Lb[k]
                                    )
                                    - 2.0
                                    * self.Lb[i]
                                    * self.Lb[i]
                                    * np.log(self.Hb[k] / self.Lb[k])
                                    - 3.0
                                    / 2.0
                                    * np.power(self.Lb[i], 4.0)
                                    * (
                                        1.0 / self.Hb[k] / self.Hb[k]
                                        - 1.0 / self.Lb[k] / self.Lb[k]
                                    )
                                )
                            )
                    else:
                        if k == i:
                            self.mp[k, i] += 1.0
                            val = dblquad(
                                _mp_int,
                                self.Lb[i],
                                self.Hb[i],
                                lambda x: self.Lb[k],
                                lambda x: self.Hb[k],
                                args=(k, i),
                            )
                            self.mp[k, i] += val / knorm[k]

                        if i < k:
                            self.mp[k, i] = (
                                4.0 * inv2[k] - 12.0 * inv4[k] * self.fb[i]
                            ) / knorm[k]

            if HAVE_PYGSL:

                def _mm_int(p, args):
                    t = args[0]
                    k = args[1]
                    i = args[2]
                    if t > p:
                        val = (
                            (4.0 / t / t - 12.0 * p * p / t / t / t / t)
                            * self.windows[k](p)
                            * self.windows[i](t)
                        )
                    else:
                        val = 0.0
                    return val

            else:

                def _mm_int(p, t, k, i):
                    if t > p:
                        return (
                            (4.0 / t / t - 12.0 * p * p / t / t / t / t)
                            * self.windows[k](p)
                            * self.windows[i](t)
                        )
                    else:
                        return 0.0

            self.mm = np.zeros((self.Nb, self.Nb))
            for k in range(self.Nb):
                # sys.stdout.write("|")
                for i in range(self.Nb):
                    if windows is None:
                        if i > k:
                            self.mm[k, i] += (
                                2.0
                                / (self.Hb[i] * self.Hb[i] - self.Lb[i] * self.Lb[i])
                                * (
                                    2.0
                                    * (
                                        self.Hb[k] * self.Hb[k]
                                        - self.Lb[k] * self.Lb[k]
                                    )
                                    * np.log(self.Hb[i] / self.Lb[i])
                                    + 3.0
                                    / 2.0
                                    * (
                                        np.power(self.Hb[k], 4.0)
                                        - np.power(self.Lb[k], 4.0)
                                    )
                                    * (
                                        1.0 / self.Hb[i] / self.Hb[i]
                                        - 1.0 / self.Lb[i] / self.Lb[i]
                                    )
                                )
                            )

                        if k == i:
                            self.mm[k, i] += 1.0
                            self.mm[k, i] += (
                                2.0
                                / (self.Hb[i] * self.Hb[i] - self.Lb[i] * self.Lb[i])
                                * (
                                    0.5
                                    * (
                                        -1.0 * self.Hb[k] * self.Hb[k]
                                        + self.Lb[k]
                                        * self.Lb[k]
                                        * (
                                            4.0
                                            - 3.0
                                            * self.Lb[k]
                                            * self.Lb[k]
                                            / self.Hb[i]
                                            / self.Hb[i]
                                            - 4.0 * np.log(self.Hb[i] / self.Lb[k])
                                        )
                                    )
                                )
                            )
                    else:
                        if k == i:
                            self.mm[k, i] += 1.0
                            val = dblquad(
                                _mm_int,
                                self.Lb[i],
                                self.Hb[i],
                                lambda x: self.Lb[k],
                                lambda x: self.Hb[k],
                                args=(k, i),
                            )
                            self.mm[k, i] += val / knorm[k]

                        if i > k:
                            self.mm[k, i] = (
                                4.0 * inv2[i] - 12.0 * inv4[i] * self.fb[k]
                            ) / knorm[k]
            # sys.stdout.write("\n")

            # compute the ell windows
            self.comp_ell_windows()

    def comp_ell_windows(self):
        # get the windows in ell
        self.have_ell_win = True

        if HAVE_PYGSL:

            def ellwin_int(theta, args):
                ell = args[0]
                win = args[1]
                n = args[2]
                return (pygsl.sf.bessel_Jn(n, ell * theta))[0] * win(theta)

        else:

            def ellwin_int(theta, ell, win, n):
                return scipy.special.jn(n, ell * theta) * win(theta)

        self.ellv = np.logspace(0.0, 5.5, 1500)
        self.ellwindowsJ0 = np.zeros((self.Nb, len(self.ellv)))
        self.ellwindowsJ4 = np.zeros((self.Nb, len(self.ellv)))
        for i in tqdm.trange(self.Nb, desc="computing BinEB ell windows", ncols=80):
            if HAVE_PYGSL:
                epsabs = 1e-6
                epsrel = 1e-6
                limit = 1000
                w = pygsl.integrate.workspace(limit)

                for j, ell in enumerate(self.ellv):
                    args = [ell, self.windows[i], 0]
                    f = pygsl.integrate.gsl_function(ellwin_int, args)
                    # code,val,err = pygsl.integrate.qag(
                    #    f,self.Lb[i],self.Hb[i],epsabs,epsrel,
                    #    limit,pygsl.integrate.GAUSS61,w
                    # )
                    code, val, err = pygsl.integrate.qags(
                        f, self.Lb[i], self.Hb[i], epsabs, epsrel, limit, w
                    )
                    self.ellwindowsJ0[i, j] = val

                for j, ell in enumerate(self.ellv):
                    args = [ell, self.windows[i], 4]
                    f = pygsl.integrate.gsl_function(ellwin_int, args)
                    # code,val,err = pygsl.integrate.qag(
                    #     f,self.Lb[i],self.Hb[i],epsabs,epsrel,limit,
                    #     pygsl.integrate.GAUSS61,w
                    # )
                    code, val, err = pygsl.integrate.qags(
                        f, self.Lb[i], self.Hb[i], epsabs, epsrel, limit, w
                    )
                    self.ellwindowsJ4[i, j] = val
            else:
                win0 = np.array(
                    [
                        (
                            scipy.integrate.quad(
                                ellwin_int,
                                self.Lb[i],
                                self.Hb[i],
                                args=(ell, self.windows[i], 0),
                                limit=100,
                            )
                        )[0]
                        for ell in self.ellv
                    ]
                )
                win4 = np.array(
                    [
                        (
                            scipy.integrate.quad(
                                ellwin_int,
                                self.Lb[i],
                                self.Hb[i],
                                args=(ell, self.windows[i], 4),
                                limit=100,
                            )
                        )[0]
                        for ell in self.ellv
                    ]
                )
                self.ellwindowsJ0[i, :] = win0
                self.ellwindowsJ4[i, :] = win4

    def write_data(self, fname):
        """
        writes a simple text file with object info

        # N L H
        100 1.0 400.0
        # Lb
        1.0 1.2 ... 398.0
        # Hb
        1.2 1.4 ... 400.0
        # fa
        1.0 1.0 .... 1.0
        # fb
        blah blah ... blah
        # fa_on
        blah blah ... blah
        # fb_on
        blah blah ... blah
        # invnorm
        blah blah ... blah
        # Mplus
        blah blah ... blah
        blah blah ... blah
        .
        .
        .
        blah blah ... blah
        # Mminus
        blah blah ... blah
        blah blah ... blah
        .
        .
        .
        blah blah ... blah
        # ellv
        blah blah ... blah
        # ellwinJ0
        blah blah ... blah
        blah blah ... blah
        .
        .
        .
        blah blah ... blah
        # ellwinJ4
        blah blah ... blah
        blah blah ... blah
        .
        .
        .
        blah blah ... blah

        """

        def write_vec(fp, vec):
            for val in vec:
                fp.write("%.20lg " % val)
            fp.write("\n#\n")

        def write_mat(fp, mat):
            shape = mat.shape
            for i in range(shape[0]):
                for val in mat[i, :]:
                    fp.write("%.20lg " % val)
                fp.write("\n")
            fp.write("#\n")

        fp = open(fname, "w")
        fp.write("# N L H\n")
        fp.write("%ld %.20lg %.20lg\n" % (self.Nb, self.L, self.H))

        fp.write("# Lb\n")
        write_vec(fp, self.Lb)
        fp.write("# Hb\n")
        write_vec(fp, self.Hb)

        fp.write("# fa\n")
        write_vec(fp, self.fa)
        fp.write("# fb\n")
        write_vec(fp, self.fb)

        fp.write("# fa_on\n")
        write_vec(fp, self.fa_on)
        fp.write("# fb_on\n")
        write_vec(fp, self.fb_on)

        fp.write("# invnorm\n")
        write_vec(fp, self.invnorm)

        fp.write("# Mplus\n")
        write_mat(fp, self.mp)
        fp.write("# Mminus\n")
        write_mat(fp, self.mm)

        fp.write("# ellv\n")
        write_vec(fp, self.ellv)
        fp.write("# ellwinJ0\n")
        write_mat(fp, self.ellwindowsJ0)
        fp.write("# ellwinJ4\n")
        write_mat(fp, self.ellwindowsJ4)

        fp.close()

    def read_data(self, fname):
        def read_vec(fp):
            line = fp.readline()
            line = line.strip()
            val = np.array([float(tag) for tag in line.split()])
            line = fp.readline()
            return val

        def read_mat(fp):
            mat = []
            line = fp.readline()
            while line[0] != "#":
                line = line.strip()
                mat.append([float(tag) for tag in line.split()])
                line = fp.readline()
            mat = np.array(mat)
            return mat

        fp = open(fname, "r")

        line = fp.readline()
        line = fp.readline()
        line = line.strip()
        line = line.split()
        self.Nb = int(line[0])
        self.L = float(line[1])
        self.H = float(line[2])

        line = fp.readline()
        self.Lb = read_vec(fp)

        line = fp.readline()
        self.Hb = read_vec(fp)

        line = fp.readline()
        self.fa = read_vec(fp)
        line = fp.readline()
        self.fb = read_vec(fp)

        line = fp.readline()
        self.fa_on = read_vec(fp)
        line = fp.readline()
        self.fb_on = read_vec(fp)

        line = fp.readline()
        self.invnorm = read_vec(fp)

        line = fp.readline()
        self.mp = read_mat(fp)
        line = fp.readline()
        self.mm = read_mat(fp)

        line = fp.readline()
        self.ellv = read_vec(fp)

        line = fp.readline()
        self.ellwindowsJ0 = read_mat(fp)
        line = fp.readline()
        self.ellwindowsJ4 = read_mat(fp)
        self.have_ell_win = True

        fp.close()

    def fplusminus(self, fptest):
        fp = fptest - np.sum(fptest * self.fa_on) * self.fa_on
        fp = fp - np.sum(fp * self.fb_on) * self.fb_on
        fm = np.dot(self.mp, fp)
        """
        code to test
        fm = np.zeros(len(fp))
        for i in range(len(fp)):
            for j in range(len(fp)):
                fm[i] += self.mp[i,j]*fp[j]
        print fm-np.dot(self.mp,fp)
        """
        return fp, fm

    def wplus(self, fp, fm):
        if not self.have_ell_win:
            self.comp_ell_windows()
        psum = np.array(
            [np.sum(self.ellwindowsJ0[:, i] * fp) for i in range(len(self.ellv))]
        )
        msum = np.array(
            [np.sum(self.ellwindowsJ4[:, i] * fm) for i in range(len(self.ellv))]
        )
        return self.ellv.copy(), (psum + msum) * 0.5

    def wminus(self, fp, fm):
        if not self.have_ell_win:
            self.comp_ell_windows()
        psum = np.array(
            [np.sum(self.ellwindowsJ0[:, i] * fp) for i in range(len(self.ellv))]
        )
        msum = np.array(
            [np.sum(self.ellwindowsJ4[:, i] * fm) for i in range(len(self.ellv))]
        )
        return self.ellv.copy(), (psum - msum) * 0.5

    def wplusminus(self, fp, fm):
        if not self.have_ell_win:
            self.comp_ell_windows()
        psum = np.array(
            [np.sum(self.ellwindowsJ0[:, i] * fp) for i in range(len(self.ellv))]
        )
        msum = np.array(
            [np.sum(self.ellwindowsJ4[:, i] * fm) for i in range(len(self.ellv))]
        )
        return self.ellv.copy(), (psum + msum) * 0.5, (psum - msum) * 0.5
