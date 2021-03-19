CSV files have been at least partially normalized. Every file should have at least the following columns.
    id: the unique id for each person
    x: x position, usually in meters
    y: y position, usually in meters
    (time|t): time, ususally in seconds

Power Law Paper/
    url: http://motion.cs.umn.edu/PowerLaw/PRL14_supplemental.pdf
    name: Supplemental material for: Universal Power Law Governing Pedestrian Interactions
    description: datasets linked in the powerlaw paper supplemental material. Many of the links in the actual paper are dead.

    seq_eth.csv/
    - url: https://icu.ee.ethz.ch/research/datsets.html
    - name: BIWI Walking Pedestrians dataset
    - description: This sequence was acquired from the top of the ETH main building, Zurich, by Stefano Pellegrini and Andreas Ess in 2009. 

    seq_hotel.ccsv/
    -url: https://icu.ee.ethz.ch/research/datsets.html
    - name: BIWI Walking Pedestrians dataset
    - description: This sequence was acquired from the 4th floor of an hotel in Bahanhofstr, Zurich, by Stefano Pellegrini and Andreas Ess in 2009.

    students003.csv/
    - url: dead
    - description: tracked trajectories of people students outside a university building

    zara01.csv and zara02.csv/
    - url: dead
    - description: tracked trajectories of people outside walking on a street

Pedestrian Dynamics Data Archive/
    url: https://ped.fz-juelich.de/da/doku.php
    description: repo of various crowd datasets. The datasets included here are a very small subset of the available datasets online.

    bottleneck/
        250_q_45_h0.csv
        - url: https://ped.fz-juelich.de/da/doku.php#bottleneck_and_social_groups
        - name: School GymBay
        - description: experiment with people leaving a room through a doorway

        150_q_56_h0.csv
        - url: https://ped.fz-juelich.de/da/doku.php#crowds_in_front_of_bottlenecks_from_the_perspective_of_physics_and_social_psychology
        - name: Crowds in front of bottlenecks from the perspective of physics and social psychology
        - description: experiment with people leaving a room through a narrow opening

    hallway/
        BI_CORR_400_A_1.csv
        - url: https://ped.fz-juelich.de/da/doku.php#corridor_bidirectional_flow
        - name: Corridor, bidirectional flow
        - description: experiment, people in hallway going in both directions

        crossing_90_a_01.csv
        - url: https://ped.fz-juelich.de/da/doku.php#crossing_90_degree_angle
        - name: Crossing, 90 degree angle
        - description: experiment, people in hallways crossing at 90 degrees

        traj_UNI_CORR_500_01.csv
        - url: https://ped.fz-juelich.de/da/doku.php#corridor_unidirectional_flow
        - name: Corridor, unidirectional flow
        - description: experiment, people in hallway all going one way