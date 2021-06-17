# cycling-tracker

### Description
 - Uses the algorithm proposed in [this paper](https://www.researchgate.net/profile/Ross_Cutler/publication/3813403_Real-time_periodic_motion_detection_analysis_and_applications/links/5969c77baca2728ca6803943/Real-time-periodic-motion-detection-analysis-and-applications.pdf) to track the cadence (i.e., the pedal RPMs) of a stationary bike
 - The cadence, the wheel radius, and the number of wheel rotations per pedal rotations (a bike-specific constant) are used to estimate current and average speed
 - Runs in realtime on a 2019 MacBook Pro
 - Basic testing (using a timer while counting number of revolutions) suggests that the RPMs are tracked with single digit-level accuracy

### TODO
 - Add docstrings
 - Print statistics to screen rather than terminal
 - More rigorous testing
