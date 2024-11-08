from src.backend.Filter.Filter import *
from src.backend.Filter.TemplateLimit import *


class GroupDelay(Filter):
    def __init__(self, ft: FilterData.ft, GD: FilterData.GD, tolerance: FilterData.tolerance, gain: FilterData.gain,
                 Nmax: FilterData.Nmax, Nmin: FilterData.Nmin, Qmax: FilterData.Qmax, Denorm: FilterData.Denorm):
        self.type = FilterType.GD
        self.reqData = {FilterData.ft: ft,
                        FilterData.GD: GD,
                        FilterData.tolerance: tolerance,
                        FilterData.gain: gain,
                        FilterData.Nmax: Nmax, FilterData.Nmin: Nmin, FilterData.Qmax: Qmax,
                        FilterData.Denorm: Denorm}
        self.default = {FilterData.ft: 10e3,            #In Hz
                        FilterData.GD: 100,             #In useg
                        FilterData.tolerance: 5,        #In %
                        FilterData.gain: 0,
                        FilterData.Nmax: None, FilterData.Nmin: None, FilterData.Qmax: None,
                        FilterData.Denorm: 0}

    def validate(self) -> (bool, str):  # Returns true and "Ok" if everything is fine, false and "ErrMsg" if not
        valid = False
        message = "Ok"
        if self.reqData[FilterData.tolerance] < 0 or self.reqData[FilterData.tolerance] > 100:
            message = "Error: Tolerance must be between 0 and 100."
        elif self.reqData[FilterData.GD] < 0:
            message = "Error: Group Delay must be positive."
        else:
            valid = True
        return valid, message

    def get_template_limits(self):  # Create one set of squares for denormalized graph, and one set for
                                    # normalized graph
        ft = self.reqData[FilterData.ft]
        tol = self.reqData[FilterData.tolerance]
        maxGD = self.reqData[FilterData.GD]*(1-tol)
        GD = self.reqData[FilterData.GD]


        denormLimit1 = Limit(Dot(0, maxGD), Dot(ft, maxGD), Dot(0, -1e9), Dot(ft, -1e9))
        denormLimit = [denormLimit1]

        normLimit1 = Limit(Dot(0, 1-tol), Dot(ft*GD*1e-6, 1-tol), Dot(0, -1e9), Dot(ft*GD*1e-6, -1e9))
        normLimit = [normLimit1]

        return [denormLimit, normLimit]
