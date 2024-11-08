from src.backend.Filter.Filter import *
from src.backend.Filter.TemplateLimit import *


class BandReject(Filter):
    def __init__(self, Aa: FilterData.Aa, faMin: FilterData.faMin, faMax: FilterData.faMax,
                 Ap: FilterData.Ap, fpMin: FilterData.fpMin, fpMax: FilterData.fpMax,
                 gain: FilterData.gain,
                 Nmax: FilterData.Nmax, Nmin: FilterData.Nmin, Qmax: FilterData.Qmax, Denorm: FilterData.Denorm):
        self.type = FilterType.BR
        self.reqData = {FilterData.Aa: Aa, FilterData.faMin: faMin, FilterData.faMax: faMax,
                        FilterData.Ap: Ap, FilterData.fpMin: fpMin, FilterData.fpMax: fpMax,
                        FilterData.gain: gain,
                        FilterData.Nmax: Nmax, FilterData.Nmin: Nmin, FilterData.Qmax: Qmax,
                        FilterData.Denorm: Denorm}
        self.default = {FilterData.Aa: 30, FilterData.faMin: 11e3, FilterData.faMax: 19e3,
                        FilterData.Ap: 5, FilterData.fpMin: 10e3, FilterData.fpMax: 20e3,
                        FilterData.gain: 0,
                        FilterData.Nmax: None, FilterData.Nmin: None, FilterData.Qmax: None,
                        FilterData.Denorm: 0}

    def validate(self) -> (bool, str):  # Returns true and "Ok" if everything is fine, false and "ErrMsg" if not
        valid = False
        message = "Ok"
        if self.reqData[FilterData.Aa] < 0 or self.reqData[FilterData.Ap] < 0:
            message = "Error: Enter positive values for Aa and Ap."
        elif self.reqData[FilterData.Aa] < self.reqData[FilterData.Ap]:
            message = "Error: Aa must be greater than Ap."
        elif self.reqData[FilterData.fpMin] > self.reqData[FilterData.faMin]:
            message = "Error: fp- must be less than fa-."
        elif self.reqData[FilterData.faMin] > self.reqData[FilterData.faMax]:
            message = "Error: fa- must be less than fa+."
        elif self.reqData[FilterData.faMax] > self.reqData[FilterData.fpMax]:
            message = "Error: fa+ must be less than fp+."
        else:
            valid = True
        return valid, message

    def get_template_limits(self):  # Create one set of squares for denormalized graph, and one set for
        # normalized graph
        Ap = self.reqData[FilterData.Ap]
        Aa = self.reqData[FilterData.Aa]
        fpMin = self.reqData[FilterData.fpMin]
        fpMax = self.reqData[FilterData.fpMin]
        faMin = self.reqData[FilterData.faMin]
        faMax = self.reqData[FilterData.faMin]

        denormLimit1 = Limit(Dot(0, 1e9), Dot(fpMin, 1e9), Dot(0, Ap), Dot(fpMin, Ap))
        denormLimit2 = Limit(Dot(faMin, Aa), Dot(faMax, Aa), Dot(faMin, 0), Dot(faMax, 0))
        denormLimit3 = Limit(Dot(fpMax, 1e9), Dot(1e12, 1e9), Dot(fpMax, Ap), Dot(1e12, Ap))
        denormLimit = [denormLimit1, denormLimit2, denormLimit3]

        selectivity = (faMax - faMin) / (fpMax - fpMin)  # K = deltafa/deltafp
        normalizedF1 = 1 / (2 * pi)
        normalizedF2 = 1 / (2 * pi * selectivity)

        normLimit1 = Limit(Dot(0, 1e9), Dot(normalizedF1, 1e9), Dot(0, Ap), Dot(normalizedF1, Ap))
        normLimit2 = Limit(Dot(normalizedF2, Aa), Dot(1e12, Aa), Dot(normalizedF2, 0), Dot(1e12, 0))
        normLimit = [normLimit1, normLimit2]

        return [denormLimit, normLimit]
