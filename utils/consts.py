"""File containing the general settings for the experiments
"""

"""
    SETTINGS: Drift
"""
DRIFT_DURATION = 60*10  # seconds => 10min
DRIFT_FREQUENCY = 10  # Hz
"""
    SETTINGS: Repeatability
"""
REPEATABILITY_DURATION = 2  # seconds
REPEATABILITY_FREQUENCY = 150  # Hz
"""
    SETTINGS: Static accuracy
"""
STATIC_ACC_DURATION = 2  # seconds
STATIC_ACC_FREQUENCY = 150  # Hz
"""
    SETTINGS: Dynamic accuarcy
"""
DYNAMIC_ACC_FREQUENCY = 150  # Hz
MOVING_THRESHOLD = 0.001  # mm distance the robot can move before pose recording starts
"""
    SETTINGS: Libsurvive
"""
LIBSURVIVE_STABILITY_THRESHOLD = 0.001  # mm distance between consecutive poses to be considered towards the stability counter
# number of consecutive poses below the threshold for libsurive sensor fusion to be considered stable
LIBSURVIVE_STABILITY_COUNTER = 100
