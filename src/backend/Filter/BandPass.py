from src.backend.Filter.Filter import *
from src.backend.Filter.TemplateLimit import *


class BandPass(Filter):
    def __init__(self, Aa: FilterData.Aa, faMin: FilterData.faMin, faMax: FilterData.faMax,
                 Ap: FilterData.Ap, fpMin: FilterData.fpMin, fpMax: FilterData.fpMax,
                 gain: FilterData.gain,
                 Nmax: FilterData.Nmax, Nmin: FilterData.Nmin, Qmax: FilterData.Qmax, Denorm: FilterData.Denorm):
        self.type = FilterType.BP
        self.reqData = {FilterData.Aa: Aa, FilterData.faMin: faMin, FilterData.faMax: faMax,
                        FilterData.Ap: Ap, FilterData.fpMin: fpMin, FilterData.fpMax: fpMax,
                        FilterData.gain: gain,
                        FilterData.Nmax: Nmax, FilterData.Nmin: Nmin, FilterData.Qmax: Qmax,
                        FilterData.Denorm: Denorm}
        self.default = {FilterData.Aa: 30, FilterData.faMin: 10e3, FilterData.faMax: 20e3,
                        FilterData.Ap: 5, FilterData.fpMin: 11e3, FilterData.fpMax: 19e3,
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
        elif self.reqData[FilterData.faMin] > self.reqData[FilterData.fpMin]:
            message = "Error: fa- must be less than fp-."
        elif self.reqData[FilterData.fpMin] > self.reqData[FilterData.fpMax]:
            message = "Error: fp- must be less than fp+."
        elif self.reqData[FilterData.fpMax] > self.reqData[FilterData.faMax]:
            message = "Error: fp+ must be less than fa+."
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

        denormLimit1 = Limit(Dot(0, Aa), Dot(faMin, Aa), Dot(0, 0), Dot(faMin, 0))
        denormLimit2 = Limit(Dot(fpMin, 1e9), Dot(fpMax, 1e9), Dot(fpMin, Ap), Dot(fpMax, Ap))
        denormLimit3 = Limit(Dot(faMax, Aa), Dot(1e12, Aa), Dot(faMax, 0), Dot(1e12, 0))
        denormLimit = [denormLimit1, denormLimit2, denormLimit3]

        selectivity = (fpMax-fpMin) / (faMax-faMin)  # K = deltafp/deltafa
        normalizedF1 = 1 / (2 * pi)
        normalizedF2 = 1 / (2 * pi * selectivity)

        normLimit1 = Limit(Dot(0, 1e9), Dot(normalizedF1, 1e9), Dot(0, Ap), Dot(normalizedF1, Ap))
        normLimit2 = Limit(Dot(normalizedF2, Aa), Dot(1e12, Aa), Dot(normalizedF2, 0), Dot(1e12, 0))
        normLimit = [normLimit1, normLimit2]

        return [denormLimit, normLimit]
