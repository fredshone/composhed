# Sources

These are the publications that define the *structure* of each variant: which
decisions are modelled, in what order, and with what kind of model. No
coefficients are taken from any of them; every model is re-estimated on NTS
2023.

> **Not yet checked in detail.** The model forms were settled in `PLAN.md` §1,
> in answer to `SPECIFICATION.md` §11. §11 asks for each row below to be checked
> against the cited document, down to section and table numbers, before the
> paper is finalised. That check has not been done.

## By variant

| Variant | What it borrows | Source |
|---|---|---|
| Compositional | Sequential, tour-based day structure | Bowman & Ben-Akiva (2001); Bhat et al. (2004) |
| Compositional | Closest architecture: stepwise regression models of activity schedules | Hilgert et al. (2017) |
| MDCEV | Joint time-allocation model | Bhat (2005); Bhat (2008) |
| MDCEV | Forecasting with the MDCEV model | Pinjari & Bhat (2021) |
| MDCEV | Estimation software | Bierlaire (2003) |
| CEMDAP | Overall chain (worker / non-worker patterns, sequential generation, continuous time) | Bhat et al. (2004) |
| CEMDAP | Hazard-based start and duration models | Bhat et al. (2004); Pinjari & Bhat (2011) |
| DaySim | Day pattern, tour priority, joint departure / return choice, half-tour stops | Bowman & Ben-Akiva (2001) |
| DaySim | Time windows, 30-minute periods | Bradley, Bowman & Griesenbeck (2010) |
| ActivitySim | Daily activity pattern (CDAP) | Bradley & Vovsha (2005) |
| ActivitySim | Tour scheduling over start × end alternatives with a timetable | Vovsha & Bradley (2004) |
| ActivitySim | Model chain, person types, tour and stop frequency, trip purpose and trip scheduling tables, calibration by segment constants | ActivitySim documentation and example model (activitysim.github.io); Vovsha, Petersen & Donnelly (2002) |
| All | Multinomial, ordered and conditional logit | Train (2009) |
| All | Accelerated failure time duration models | Kalbfleisch & Prentice (2002) |

## References

- Bhat, C.R. (2005). A multiple discrete-continuous extreme value model: formulation and application to discretionary time-use decisions. *Transportation Research Part B*, 39(8), 679–707.
- Bhat, C.R. (2008). The multiple discrete-continuous extreme value (MDCEV) model: role of utility function parameters, identification considerations, and model extensions. *Transportation Research Part B*, 42(3), 274–303.
- Bhat, C.R., Guo, J.Y., Srinivasan, S. & Sivakumar, A. (2004). Comprehensive econometric microsimulator for daily activity-travel patterns. *Transportation Research Record*, 1894, 57–66.
- Bierlaire, M. (2003). BIOGEME: a free package for the estimation of discrete choice models. *Proceedings of the 3rd Swiss Transportation Research Conference (STRC)*, Ascona, Switzerland.
- Bowman, J.L. & Ben-Akiva, M.E. (2001). Activity-based disaggregate travel demand model system with activity schedules. *Transportation Research Part A*, 35(1), 1–28.
- Bradley, M., Bowman, J.L. & Griesenbeck, B. (2010). SACSIM: An applied activity-based model system with fine-level spatial and temporal resolution. *Journal of Choice Modelling*, 3(1), 5–31.
- Bradley, M. & Vovsha, P. (2005). A model for joint choice of daily activity pattern types of household members. *Transportation*, 32(5), 545–571.
- Hilgert, T., Heilig, M., Kagerbauer, M. & Vortisch, P. (2017). Modeling week activity schedules for travel demand models. *Transportation Research Record*, 2666, 69–77.
- Kalbfleisch, J.D. & Prentice, R.L. (2002). *The Statistical Analysis of Failure Time Data*, 2nd ed. Wiley.
- Pinjari, A.R. & Bhat, C.R. (2011). Activity-based travel demand analysis. In de Palma, A., Lindsey, R., Quinet, E. & Vickerman, R. (eds.), *A Handbook of Transport Economics*. Edward Elgar.
- Pinjari, A.R. & Bhat, C.R. (2021). Computationally efficient forecasting procedures for Kuhn-Tucker consumer demand model systems: application to residential energy consumption analysis. *Journal of Choice Modelling*, 39, 100283.
- Train, K. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge University Press.
- Vovsha, P. & Bradley, M. (2004). Hybrid discrete choice departure-time and duration model for scheduling travel tours. *Transportation Research Record*, 1894, 46–56.
- Vovsha, P., Petersen, E. & Donnelly, R. (2002). Microsimulation in travel demand modeling: lessons learned from the New York best practice model. *Transportation Research Record*, 1805, 68–77.
