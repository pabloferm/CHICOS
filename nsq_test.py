import nuSQuIDS as nsq
import numpy as np

nuSQ = nsq.nuSQUIDS(3,nsq.NeutrinoType.neutrino)
units = nsq.Const()

nuSQ.Set_MixingAngle(0,1, np.radians(33.44)) # sets \theta_{12} in radians.
nuSQ.Set_MixingAngle(0,2, np.radians(8.57)) # sets \theta_{13}} in radians.
nuSQ.Set_MixingAngle(1,2, np.radians(49.2)) # sets \theta_{23}} in radians.
nuSQ.Set_SquareMassDifference(1, 7.42e-5) # sets dm^2_{21} in eV^2.
nuSQ.Set_SquareMassDifference(2, 2.51e-3) # sets dm^2_{31} in eV^2.
nuSQ.Set_CPPhase(0,2, np.radians(234.))

nuSQ.Set_Body(nsq.ConstantDensity(2.8,0.5))

nuSQ.Set_rel_error(1.0e-15)
nuSQ.Set_abs_error(1.0e-17)

energies = np.geomspace(5e-2, 5e1, 100)  # GeV
baselines = np.geomspace(1e-1, 1e4, 100) # km

print("E(GeV),L(km),P(e->e),P(mu->e),P(tau->e),P(e->mu),P(mu->mu),P(tau->mu),P(e->tau),P(mu->tau),P(tau->tau)")

for Enu in energies:
    for L in baselines:
        nuSQ.Set_Track(nsq.ConstantDensity.Track(L*units.km))
        nuSQ.Set_E(Enu*units.GeV)
        nuSQ.Set_initial_state(np.array([1.,0.,0.]),nsq.Basis.flavor)
        nuSQ.EvolveState()
        ee = nuSQ.EvalFlavor(0)
        emu = nuSQ.EvalFlavor(1)
        etau = nuSQ.EvalFlavor(2)
        nuSQ.Set_initial_state(np.array([0.,1.,0.]),nsq.Basis.flavor)
        nuSQ.EvolveState()
        mue = nuSQ.EvalFlavor(0)
        mumu = nuSQ.EvalFlavor(1)
        mutau = nuSQ.EvalFlavor(2)
        nuSQ.Set_initial_state(np.array([0.,0.,1.]),nsq.Basis.flavor)
        nuSQ.EvolveState()
        taue = nuSQ.EvalFlavor(0)
        taumu = nuSQ.EvalFlavor(1)
        tautau = nuSQ.EvalFlavor(2)
        print(f"{Enu},{L},{ee},{mue},{taue},{emu},{mumu},{taumu},{etau},{mutau},{tautau}")

