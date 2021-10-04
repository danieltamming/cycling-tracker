# cycling-tracker

### Description
 - Uses the algorithm proposed in [this paper](https://www.researchgate.net/profile/Ross_Cutler/publication/3813403_Real-time_periodic_motion_detection_analysis_and_applications/links/5969c77baca2728ca6803943/Real-time-periodic-motion-detection-analysis-and-applications.pdf) to track the cadence (i.e., the pedal RPMs) of a stationary bike
 - The cadence, the wheel radius, and the number of wheel rotations per pedal rotations (a bike-specific constant) are used to estimate current and average speed
 - Current and average speed and RPMs are printed to the command line
 - As program exits, a graph of RPMs over time is produced and all other data is saved to a csv file
 - Runs in real-time on a 2019 MacBook Pro
 - Basic testing (using a timer while counting number of revolutions) suggests that the RPMs are tracked with single digit-level accuracy

### How to Use

#### Quick Start
Place your laptop next to your stationary bike and make sure the bike pedals are in full view of the laptop camera. 
```
git clone https://github.com/danieltamming/cycling-tracker
cd cycling-tracker
conda env create -f environment.yml
python cyclingtracker.py
```
When finished, enter Control-C. The program will then display your RPMs over time.

#### Bike-Specific Parameters
To get the most accurate results, the following two parameters should be adjusted to fit the user's bike. 

 - `WHEEL_RADIUS`, the radius of the stationary bike's front wheel, in meters. Default = 0.22
 - `WHEELS_PER_PEDAL`, the number of wheel revolutions per pedal revolutions. Default = 3.5

### Front-End (In Development On The *dev* Branch)
 - A Plotly dashboard that gives a real-time graphical display of RPMs and speed
 - The dashboard successfully runs while processing a pre-recorded video, but has not yet been tested on real-time video

### Other Next Steps
 - Add docstrings
 - More rigorous parameter tuning and testing, for example:
   - Using a cadence tracking sensor, rather than a stopwatch, to determine the ground truth RPMs
   - Varying the window size and sample rates to optimize accuracy while maintaining the ability to run in real-time