from .base import SimpleModel

models_dict =\
{
    "default": SimpleModel,
    # "random": RandomModel,
    # "checkpoint": CheckpointModel,
    # "mcdropout": MCDropoutModel,
    # "tt_approx": TestTimeOnly_ApproximateDropoutModel,
    # "naive-ensemble": NaiveEnsembleModel,
    # "naive-ensemble-sum": NaiveEnsembleSummationModel,
    # "naive-ensemble-sum-head": NaiveEnsembleSummationModel_HeadOnly,
    # "kernel-ensemble": NaiveEnsembleModel_KernelNoiseOnly,
    # "kernel-ensemble-sum": NaiveEnsembleSummationModel_KernelNoiseOnly,
    # "naive-dropout": NaiveDropoutModel,
    # "train-diff": TestTimeOnly_GroundTruthInit_PlainTrainingDifference,

    # "wasserstein-GTinit": TestTimeOnly_GroundTruthInit_WassersteinModel,
    # "wasserstein-GaussianInit": TestTimeOnly_GaussianInit_WassersteinModel,

    # "velocity-std": TestTimeOnly_GroundTruthInit_VelocityModel,

    # "hessian-variance": TestTimeOnly_HessianVariance,
    # "hessian-variance-negate": TestTimeOnly_HessianVariance_Negate,

    # "plain-ntk": TestTimeOnly_NTK,

    # "plain-ntk-withNTKinv": TestTimeOnly_NTK_withInv,
    # "plain-ntk-init": TestTimeOnly_NTK_initialization,
    # "plain-ntk-zero-init": TestTimeOnly_NTK_zero_initialization,
    # "plain-ntk-init-cls": TestTimeOnly_NTK_initialization_classification,
    # "plain-ntk-zero-init-cls": TestTimeOnly_NTK_zero_initialization_classification,
    # "plain-ntk-zero-init-cls-nosmax": TestTimeOnly_NTK_zero_initialization_classification_nosoftmax,
    # "plain-ntk-zero-init-mean": TestTimeOnly_NTK_zero_initialization_mean,

    # "plain-ntk-zero-init-simple-multidim": TestTimeOnly_NTK_zero_initialization_simple_multidim,

    # "la": LaplaceRedux,

    # "inject-test": uq.InjectTest,
    # "inject-test-fluc": uq.InjectTest_Fluc,
    # "inject-test-fluc-normed": uq.InjectTest_NormalizedFluc,
    # "inject-test-subtract": uq.InjectTest_Subtract,
    # "inject-test-indepdet": uq.InjectTest_IndepDet
}
