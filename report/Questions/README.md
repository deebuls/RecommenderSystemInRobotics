# Questions From the paper

## features : 
 * Is there a relation of features with PDDL (specifically Predicates of PDDL)?

In this work, we do not consider high-level action models based on
PDDL. We did that in a previous work (ICRA13). There, you can
represent each learned condition with a logical predicate.
 * Examples of features and Actions from the paper.

| Features | Actions |
|----------|---------|
| Feature describing pose of the gripper and the grasped object relative to the target object.| Placing a grasped object on another object |
| feature describing pose of the gripper relative to target.| Reaching for a specific object |
| feature describing pose of the cup relative to cup and fork.| Placing a grasped cup next to plate and fork |

* Question : How do we map between this feature and action ? How to generate the action from 1 feature ? Is this action related to the action of PDDL ?
An action is nothing but the collection of all feature trajectories
observed in the demonstrations. The point is that a certain template
will tell you which features are more relevant than others so you only
use those when modeling the action. We also mainly focus on the
beginning and end of each trajectory and use those to model the
bivariate histogram for each feature dimension.

## Expert 'e' :
 * Please can you explain experts with respect to the given examples.
     * How many experts were used ?
     * what values did they suggest ?
I believe in these experiments we had three experts. Each one was
responsible for describing some "category" of actions (actions where
relative poses are important, actions where absolute poses are
important, etc).
In reality though, the number of experts is not too important. At
least there is room for exploring the behavior of the system with
respect to that. You should start with one expert (you) and just set
some templates from common manipulation actions and see which ones the
system recommends. That should be sufficient.i

## Recommender System :
### General user of recommender in movie recommendation :

|n | Users | Genres |
|-|--------|---------|
|Movies| ..| ..|
    
    Based what the user has rated and the genres of those movies the recommender needs to predict the preference of the user
### Recommender system in  model

|n | Demonstration | Experts|
|--|--------|---------|
| features | ..     | ..  |
    
    Based on what each Demonstration the values given to the features and the experts  the recommender predicts which features to recommend 
 * Is my understanding of the recommender correct ?
 * If yes can you explain with an example what values goes in the table ?

# Implementation Details 

Since my Experimentation are to be conducted on youBot, I have some questions
regarding the Implementation.

* What did you use to record the positions of the PR2 (rosrecord ) ?
Yes, I was recording the gripper pose during the demonstrations, and
used AR-tags attached to the objects to record their poses. After that
you can compute all feature combinations by transforming poses from
one frame (e.g. object 3) to another (e.g. left gripper)

* Do you use PDDL ?
Not in this work. It's a possible extension.

* Do you use DMP for trajectory generation ?
* Once the relavant feature is selected what software did you use to generate
the trajectory ?
No DMPs in this work. Look at previous works. In this paper, we were
more interested in selecting the relevant feature dimensions. The
"software" is basically to transform the trajectories to a specific
frame (depending on the template) and then we used a controller to
execute the motion on the PR2.

For your project, I would not try to replicate every small detail of
the paper (number of experts, etc). Just get the main idea and see
what works for your specific problem. I would recommend you to read
about content-based recommender systems in general to get a better
intuition and to think what you need for your specific problem.

