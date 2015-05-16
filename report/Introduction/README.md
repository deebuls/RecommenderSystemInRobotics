# Learning skill for Mobile manipulators  by Learning by Demonstration.

Task level programming is an important research statement for mobile
manipulators in the industrial environments. From extensive analysis of
industrial robots [1], it has been observed that many
tasks in the industrial environments can be solved using pre defined building
blocks, called as skills.

In [2] an overview is presented on how to implement a skill-based
architecture, enabling reuse of skills for different industri-
al applications, programmed by shop-floor workers. It further proposes finding 
the skills by analyzing real-world implementations and Standard Operating 
Procedures (SOP) from an industrial partner. The identifyed skills has to be
consistent with human intutition. Based on the analysis by Bøgh et al [1] the
 skills identifyied for solving industrial tasks is presented in table.

|Skill      |Description            |
|:----------|:----------------------|
|Move to      |To go from one location (station) to another in the factory|
|Locate     |To determine or specify the position of anobject by searching or examining|
| Pick up     |To take hold of and lift up|
| Place     |To arrange something in a certain spot orposition|
| Unload     |Unload a container: to remove, dischargeor empty the contents from a container|
| Shovel     |To take up and move objects with a shovel|
| Check     |Quality control: the act or process of test-ing, verifying or examining|
| Align     |To make an object come into precise ad-justment or correct relative position to an-other object|
| Open     |To move (as a door) from a closed positionand make available for entry, passage oraccessible|
| Close     |To move (as a door) from an open position|
| Press     |To press against with force in order todrive or impel|
| Release     |To let go or set free from restraint e.g. release a button|
| Turn     |To turn a knob or handle|


As we can see this short set of skills is sifficient for solving a large set of
industrial tasks.

Currently much of the robot programming is carried out by engineers and are
usually written from scratch for new robots.

Learning by Demonstration has been active field of research for the past 20
years. Different approaches for learning has been explored like probabilistic
approach GMM,HMM and Dynamical level learning like DMP. But most of the learning are 
generall done at the motion primitive level. Most of the learning is to learn a
single skill. Very less efforts have been seen at learning at the skill level.

Since the skills required in the industrial environments are well defined we
can use this information for teaching the robot these skills for new scenarios 
with programming by demonstration.

## Problem Statement

* To create a set of templates for each skill. The template determine the most
important feature required for learning the skill from the deomstration.
This process of creating template is done before learning. 
* Demonstrations of the skill are done with the robot using
teleoperations.
* Based on the demonstration and the insight of which skill is being learnt the
learning algorithm recommends the best suited template for reproducing the
skill.
* The recommendations are based on the concepts of content based recommender
  systems as demonstrated by Nicholo Abdo et al [3]

[1] Bøgh, S., Hvilshøj, M., Kristiansen, M., & Madsen, O. (2012). Identifying
and evaluating suitable tasks for autonomous industrial mobile manipulators
(AIMM). The International Journal of Advanced Manufacturing Technology,
61(5-8), 713–726. http://doi.org/10.1007/s00170-011-3718-3

[2] Bøgh, S., Nielsen, O. S., Pedersen, M. R., Krüger, V., & Madsen, O. (2012).
Does your robot have skills? In The 43rd Intl. Symp. on Robotics (ISR2012).

[3] Abdo, N., Spinello, L., Burgard, W., & Stachniss, C. (n.d.). Inferring What
to Imitate in Manipulation Actions by Using a Recommender System. In ICRA (Vol.
31, pp. 189–206). Springer$}$. 

