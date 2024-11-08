from src.backend.Filter.Filter import *
from scipy import signal
from scipy import special
import numpy as np

class Legendre(object):
    def __init__(self, filter: Filter):
        self.filter = filter
        self.type = self.filter.get_type()
        self.order = None
        self.fo = None
        self.Bw = None
        self.z = None
        self.p = None
        self.k = None
        self.num = None
        self.den = None
        self.sos = None
        self.wan = None
        self.epsilon2 = None
        self.w_bode = None
        self.mag = None
        self.pha = None
        self.w_att = None
        self.A = None
        self.w_natt = None
        self.A_n = None
        self.w_tf = None
        self.h = None
        self.w_tfn = None
        self.h_n = None
        self.wgd = None
        self.GroupDelay = None
        self.timp = None
        self.impresp = None
        self.tstep = None
        self.stepresp = None
        self.calculate()

    def calc_Order(self):
        val, msg = self.filter.validate()
        if val is False:
            return msg

        fpMin = self.filter.reqData[FilterData.fpMin]
        fpMax = self.filter.reqData[FilterData.fpMax]
        Ap = self.filter.reqData[FilterData.Ap]
        if Ap == 0:
            Ap = 1e-9

        faMin = self.filter.reqData[FilterData.faMin]
        faMax = self.filter.reqData[FilterData.faMax]
        Aa = self.filter.reqData[FilterData.Aa]

        Nmin = self.filter.reqData[FilterData.Nmin]
        Nmax = self.filter.reqData[FilterData.Nmax]

        self.epsilon2 = 10 ** (Ap / 10) - 1
        order = 1

        if self.type == "Low Pass":
            self.wan = faMin / fpMin
            while self.get_L_Poly_Value(order, self.wan) < (10 ** (Aa / 10) - 1) / self.epsilon2:
                order = order + 1
            if Nmin is not None and Nmin > order:
                self.order = Nmin
            elif Nmax is not None and Nmax < order:
                self.order = Nmax
            else:
                self.order = order
        elif self.type == "High Pass":
            self.wan = fpMin / faMin
            while self.get_L_Poly_Value(order, self.wan) < (10 ** (Aa / 10) - 1) / self.epsilon2:
                order = order + 1
            if Nmin is not None and Nmin > order:
                self.order = Nmin
            elif Nmax is not None and Nmax < order:
                self.order = Nmax
            else:
                self.order = order
        elif self.type == "Band Pass":
            self.wan = np.minimum(fpMin / faMin, faMax / fpMax)
            while self.get_L_Poly_Value(order, self.wan) < (10 ** (Aa / 10) - 1) / self.epsilon2:
                order = order + 1
            if Nmin is not None and Nmin > order:
                self.order = Nmin
            elif Nmax is not None and order > Nmax:
                self.order = Nmax
            else:
                self.order = order
        elif self.type == "Band Reject":
            self.wan = np.minimum(faMin / fpMin, fpMax / faMax)
            while self.get_L_Poly_Value(order, self.wan) < (10 ** (Aa / 10) - 1) / self.epsilon2:
                order = order + 1
            if Nmin is not None and Nmin > order:
                self.order = Nmin
            elif Nmax is not None and order > Nmax:
                self.order = Nmax
            else:
                self.order = order
        else:
            message = "Error: Enter Filter Type."
            return message

    def calc_fo(self):
        val, msg = self.filter.validate()
        if val is False:
            return msg

        fpMin = self.filter.reqData[FilterData.fpMin]
        fpMax = self.filter.reqData[FilterData.fpMax]

        Denorm = self.filter.reqData[FilterData.Denorm]

        if self.type == "Low Pass":
            wo1 = self.get_L_wo()
            wo2 = wo1 * self.wan / self.get_L_wa()
            fod = 10 ** (np.log10(wo1) * (1 - Denorm / 100) + np.log10(wo2) * (Denorm / 100)) #/ (2 * np.pi)
            self.fo = fod * fpMin
        elif self.type == "High Pass":
            wo1 = self.get_L_wo()
            wo2 = wo1 * self.wan / self.get_L_wa()
            fod = 10 ** (np.log10(wo1) * (Denorm / 100) + np.log10(wo2) * (1 - Denorm / 100)) #/ (2 * np.pi)
            self.fo = fpMin / fod
        elif self.type == "Band Pass":
            Bw = 2 * np.pi * (fpMax - fpMin)
            wp = 2 * np.pi * np.sqrt(fpMin * fpMax)
            wo1 = self.get_L_wo()
            wo2 = wo1 * self.wan / self.get_L_wa()
            wod = 10 ** (np.log10(wo1) * (1 - Denorm / 100) + np.log10(wo2) * Denorm / 100)
            wo1 = (np.sqrt((wod * Bw) ** 2 + 4 * wp ** 2) + wod * Bw) / 2
            wo2 = (np.sqrt((wod * Bw) ** 2 + 4 * wp ** 2) - wod * Bw) / 2
            self.fo = np.sqrt(wo1 * wo2) / (2 * np.pi)
            self.Bw = abs(wo1 - wo2) / (2 * np.pi)
        elif self.type == "Band Reject":
            Bw = 2 * np.pi * (fpMax - fpMin)
            wp = 2 * np.pi * np.sqrt(fpMin * fpMax)
            wo1 = self.get_L_wo()
            wo2 = wo1 * self.wan / self.get_L_wa()
            wod = 10 ** (np.log10(wo1) * (1 - Denorm / 100) + np.log10(wo2) * Denorm / 100)
            wo1 = (np.sqrt((Bw / wod) ** 2 + 4 * wp ** 2) + Bw / wod) / 2
            wo2 = (np.sqrt((Bw / wod) ** 2 + 4 * wp ** 2) - Bw / wod) / 2
            self.fo = np.sqrt(wo1 * wo2) / (2 * np.pi)
            self.Bw = abs(wo1 - wo2) / (2 * np.pi)
        else:
            message = "Error: Enter Filter Type."
            return message

    def calc_Denormalization_zpk(self):
        if self.type == "Low Pass":
            z, p, k = self.get_L_zpk()
            self.z, self.p, self.k = signal.lp2lp_zpk(z, p, k, wo=2 * np.pi * self.fo)
        elif self.type == "High Pass":
            z, p, k = self.get_L_zpk()
            self.z, self.p, self.k = signal.lp2hp_zpk(z, p, k, wo=2 * np.pi * self.fo)
        elif self.type == "Band Pass":
            z, p, k = self.get_L_zpk()
            self.z, self.p, self.k = signal.lp2bp_zpk(z, p, k, wo=2 * np.pi * self.fo, bw=2 * np.pi * self.Bw)
        elif self.type == "Band Reject":
            z, p, k = self.get_L_zpk()
            self.z, self.p, self.k = signal.lp2bs_zpk(z, p, k, wo=2 * np.pi * self.fo, bw=2 * np.pi * self.Bw)

    def check_Q(self) -> bool:
        Qmax = self.filter.reqData[FilterData.Qmax]
        if self.order > 1 and Qmax is not None:
            z, p, k = self.get_zpk()
            q_arr = []
            for pole in p:
                q = abs(abs(pole) / (2 * pole.real))
                q_arr.append(q)
            q_sys = np.max(q_arr)
            if q_sys > Qmax and self.order > 1:
                self.order = self.order - 1
                return False
            else:
                return True
        else:
            return True

    def get_Gain(self):
        return self.filter.reqData[FilterData.gain]

    def calc_TransFunc(self):
        val, msg = self.filter.validate()
        if val is False:
            return msg
        z, p, k = self.get_zpk()
        if self.type == "Low Pass":
            sys = signal.lti(z, p, k)
            self.w_tf, self.h = sys.freqresp(w=np.logspace(-1, 9, num=100000))
        elif self.type == "High Pass":
            sys = signal.lti(z, p, k)
            self.w_tf, self.h = sys.freqresp(w=np.logspace(-1, 9, num=100000))
        elif self.type == "Band Pass":
            sys = signal.lti(z, p, k)
            self.w_tf, self.h = sys.freqresp(w=np.logspace(-1, 9, num=100000))
        elif self.type == "Band Reject":
            sys = signal.lti(z, p, k)
            self.w_tf, self.h = sys.freqresp(w=np.logspace(-1, 9, num=100000))
        else:
            message = "Error: Enter Filter Type."
            return message

    def calc_Norm_TransFunc(self):
        z, p, k = self.get_L_zpk()
        z, p, k = signal.lp2lp_zpk(z, p, k, wo=2 * np.pi * self.fo)
        sys = signal.lti(z, p, k)
        self.w_tfn, self.h_n = sys.freqresp(w=np.logspace(-1, 9, num=100000))

    def calc_MagAndPhase(self):                                     # return angular frequency, Mag and Phase
        val, msg = self.filter.validate()
        if val is False:
            return msg
        z, p, k = self.get_zpk()
        if self.type == "Low Pass":
            sys = signal.ZerosPolesGain(z, p, k)
            self.w_bode, self.mag, self.pha = signal.bode(sys, w=np.logspace(-1, 9, num=100000))
        elif self.type == "High Pass":
            sys = signal.ZerosPolesGain(z, p, k)
            self.w_bode, self.mag, self.pha = signal.bode(sys, w=np.logspace(-1, 9, num=100000))
        elif self.type == "Band Pass":
            sys = signal.ZerosPolesGain(z, p, k)
            self.w_bode, self.mag, self.pha = signal.bode(sys, w=np.logspace(-1, 9, num=100000))
        elif self.type == "Band Reject":
            sys = signal.ZerosPolesGain(z, p, k)
            self.w_bode, self.mag, self.pha = signal.bode(sys, w=np.logspace(-1, 9, num=100000))
        else:
            message = "Error: Enter Filter Type."
            return message

    def calc_Attenuation(self):
        w, h = self.get_TransFuncWithoutGain()
        A = []
        for i in range(len(h)):
            A.append(20 * log10(abs(1 / h[i])))
        self.w_att = w
        self.A = A

    def calc_Norm_Attenuation(self):
        w, h = self.get_Norm_TransFunc()
        wn = np.divide(w, 2 * np.pi * self. fo)
        An = []
        for i in range(len(h)):
            An.append(20 * log10(abs(1 / h[i])))
        self.w_natt = wn
        self.A_n = An

    def calc_Group_Delay(self):
        w, mag, pha = self.get_MagAndPhaseWithoutGain()
        gd = np.divide(- np.diff(pha), np.diff(w))
        gd = gd.tolist()
        gd.append(gd[len(gd) - 1])
        self.GroupDelay = gd
        w = w.tolist()
        self.wgd = w

    def calc_Impulse_Response(self):
        t, out = signal.impulse(self.get_lti(), N=100000)
        self.timp = t
        self.impresp = out

    def calc_Step_Response(self):
        t, out = signal.step(self.get_lti(), N=100000)
        self.tstep = t
        self.stepresp = out

    def get_lti(self):
        z, p, k = self.get_zpGk()
        return signal.lti(z, p, k)

    def get_Norm_TransFunc(self):
        return self.w_tfn, self.h_n

    def calculate(self):
        self.calc_Order()
        self.calc_fo()
        self.calc_Denormalization_zpk()
        while self.check_Q() is False:
            self.calc_fo()
            self.calc_Denormalization_zpk()
        self.calc_TransFunc()
        self.calc_Norm_TransFunc()
        self.calc_MagAndPhase()
        self.calc_Group_Delay()
        self.calc_Impulse_Response()
        self.calc_Step_Response()
        self.calc_Attenuation()
        self.calc_Norm_Attenuation()

    #####################
    #       ALEX        #
    #####################

    def get_zpk(self):
        return self.z, self.p, self.k

    def get_zpGk(self):
        Gk = self.k * 10 ** (self.get_Gain() / 20)
        return self.z, self.p, Gk

    def get_TransFuncWithGain(self):
        hg = self.h
        gain = self.get_Gain()
        for i in range(0, len(hg)):
            hg[i] = hg[i] * 10 ** (gain / 20)
        return self.w_tf, hg

    def get_TransFuncWithoutGain(self):
        return self.w_tf, self.h

    def get_MagAndPhaseWithGain(self):
        magg = self.mag
        gain = self.get_Gain()
        for i in range(0, len(magg)):
            magg[i] = magg[i] + gain
        return self.w_bode, magg, self.pha

    def get_MagAndPhaseWithoutGain(self):
        return self.w_bode, self.mag, self.pha

    def get_Group_Delay(self):
        return self.wgd, self.GroupDelay

    def get_Attenuation(self):
        return self.w_att, self.A

    def get_Norm_Attenuation(self):
        return self.w_natt, self.A_n

    def get_Order(self):
        return self.order

    def get_very_useful_data(self):  # colocar TAL CUAL en las otras aprox
        fpMin = self.filter.reqData[FilterData.fpMin]
        fpMax = self.filter.reqData[FilterData.fpMax]
        Ap = self.filter.reqData[FilterData.Ap]

        faMin = self.filter.reqData[FilterData.faMin]
        faMax = self.filter.reqData[FilterData.faMax]
        Aa = self.filter.reqData[FilterData.Aa]

        # Nmin = self.filter.reqData[FilterData.Nmin]
        # Nmax = self.filter.reqData[FilterData.Nmax]
        return fpMin, fpMax, Ap, faMin, faMax, Aa

    def get_wan(self):
        return self.wan

    def get_Qs(self):
        z, p, k = self.get_zpk()
        q_arr = []
        for pole in p:
            q = abs(abs(pole) / (2 * pole.real))
            q_arr.append(q)
        return q_arr

    def get_Impulse_Response(self):
        return self.timp, self.impresp

    def get_Step_Response(self):
        return self.tstep, self.stepresp

    #############################
    #       Legendre Calc       #
    #############################

    def get_Legendre_Poly(self, order):
        return special.legendre(order)

    def get_even_L_Optimum(self, n):
        k = (n - 2) / 2
        if k % 2 == 0:
            a0 = 1 / (np.sqrt((k + 1) * (k + 2)))
            coefs = [a0]
            i = 1
            while i <= k:
                if i % 2 == 0:
                    coefs.append((2 * i + 1) * a0)

                else:
                    coefs.append(0)
                i += 1

            poly_sum = coefs[0] * self.get_Legendre_Poly(0)
            i = 1
            while i <= k:
                poly_sum = np.polyadd(poly_sum, coefs[i] * self.get_Legendre_Poly(i))
                i += 1

            dpoly = np.polymul(poly_sum, poly_sum)
            dpoly = np.polymul(dpoly, np.poly1d([1, 1]))
            poly = np.polyint(dpoly)
            uplim = np.poly1d([2, 0, -1])
            lowlim = -1
            poly = np.polysub(np.polyval(poly, uplim), np.polyval(poly, lowlim))

        else:
            a1 = 3 / (np.sqrt((k + 1) * (k + 2)))
            coefs = [0, a1]
            i = 2
            while i <= k:
                if i % 2 == 0:
                    coefs.append(0)
                else:
                    coefs.append((2 * i + 1) * a1 / 3)
                i += 1

            poly_sum = coefs[0] * self.get_Legendre_Poly(0)
            i = 1
            while i <= k:
                poly_sum = np.polyadd(poly_sum, coefs[i] * self.get_Legendre_Poly(i))
                i += 1

            dpoly = np.polymul(poly_sum, poly_sum)
            dpoly = np.polymul(dpoly, np.poly1d([1, 1]))
            poly = np.polyint(dpoly)
            uplim = np.poly1d([2, 0, -1])
            lowlim = -1
            poly = np.polysub(np.polyval(poly, uplim), np.polyval(poly, lowlim))
        return poly

    def get_odd_L_Optimum(self, n):
        k = (n - 1) / 2
        ao = 1 / (np.sqrt(2) * (k + 1))
        coefs = [ao]
        i = 1
        while i <= k:
            coefs.append((2 * i + 1) * ao)
            i += 1

        poly_sum = coefs[0] * self.get_Legendre_Poly(0)
        i = 1
        while i <= k:
            poly_sum = np.polyadd(poly_sum, coefs[i] * self.get_Legendre_Poly(i))
            i += 1

        dpoly = np.polymul(poly_sum, poly_sum)
        poly = np.polyint(dpoly)
        uplim = np.poly1d([2, 0, -1])
        lowlim = -1
        poly = np.polysub(np.polyval(poly, uplim), np.polyval(poly, lowlim))
        return poly

    def get_L_Poly(self, n):
        if n % 2 == 0:
            Ln = self.get_even_L_Optimum(n)
        else:
            Ln = self.get_odd_L_Optimum(n)
        return Ln

    def get_L_Poly_Value(self, n, wan):
        value = np.polyval(self.get_L_Poly(n), wan)
        return value

    def get_L_Filter_Poly(self):
        Ln = self.get_L_Poly(self.order)
        e2Ln = np.polymul(np.poly1d([self.epsilon2]), Ln)
        A = np.polyadd(np.poly1d([1]), e2Ln)
        return A

    def get_L_zpk(self):
        z = []
        r = 1j * np.roots(self.get_L_Filter_Poly())
        p = []
        for i in range(0, len(r)):
            if r[i].real < 0:
                p.append(r[i])
        k = np.polyval(self.get_L_Filter_Poly(), 0)
        for pole in p:
            k *= abs(pole)
        return z, p, k

    def get_L_System(self):
        z, p, k = self.get_L_zpk()
        return signal.lti(z, p, k)

    def get_L_wo(self):
        sys = self.get_L_System()
        w, mag, pha = signal.bode(sys, w=np.logspace(-1, 3, num=1000))
        for i in range(len(mag)):
            if mag[i] <= -self.filter.reqData[FilterData.Ap]:
                wo = w[i]
                return wo

    def get_L_wa(self):
        sys = self.get_L_System()
        w, mag, pha = signal.bode(sys, w=np.logspace(-1, 3, num=1000))
        for i in range(len(mag)):
            if mag[i] <= -self.filter.reqData[FilterData.Aa]:
                wa = w[i]
                return wa
