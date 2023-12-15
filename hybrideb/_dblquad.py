#!/usr/bin/env python
import scipy.integrate

__all__ = ["dblquad", "dblquad_pygsl", "dblquad_scipy", "HAVE_PYGSL"]

HAVE_PYGSL = False
try:
    import pygsl.integrate
    import pygsl.sf

    HAVE_PYGSL = True
except ImportError:
    pass

if HAVE_PYGSL:

    def dblquad_pygsl(
        func, a, b, gfun, hfun, epsabs=1e-6, epsrel=1e-6, limit=100, args=[]
    ):
        def _infunc(x, p):
            func = p[0]
            gfun = p[1]
            hfun = p[2]
            epsrel = p[3]
            epsabs = p[4]
            limit = p[5]
            args = p[6:]
            w = pygsl.integrate.workspace(limit)
            a = gfun(x)
            b = hfun(x)
            pi = [x]
            pi = pi + args
            f = pygsl.integrate.gsl_function(func, pi)
            code, val, err = pygsl.integrate.qag(
                f, a, b, epsabs, epsrel, limit, pygsl.integrate.GAUSS61, w
            )
            return val

        w = pygsl.integrate.workspace(limit)
        p = [func, gfun, hfun, epsrel, epsabs, limit]
        p += args
        f = pygsl.integrate.gsl_function(_infunc, p)
        code, val, err = pygsl.integrate.qag(
            f, a, b, epsabs, epsrel, limit, pygsl.integrate.GAUSS61, w
        )
        return val

    dblquad = dblquad_pygsl
else:

    def dblquad_scipy(
        func, a, b, gfun, hfun, args=(), epsrel=1e-12, epsabs=1e-12, limit=100
    ):
        def _infunc(x, func, gfun, hfun, more_args):
            a = gfun(x)
            b = hfun(x)
            myargs = (x,) + tuple(more_args)
            myargs = (myargs,)
            return scipy.integrate.quad(
                func, a, b, args=myargs, epsrel=epsrel, epsabs=epsabs, limit=limit
            )[0]

        val, err = scipy.integrate.quad(
            _infunc,
            a,
            b,
            (func, gfun, hfun, args),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        return val

    dblquad = dblquad_scipy
