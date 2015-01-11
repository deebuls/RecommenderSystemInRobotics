# Questions From the paper

## features : 
 * Is there a relation of features with PDDL (specifically Predicates of PDDL)?
 * Examples of features and Actions from the paper.

| Features | Actions |
|----------|---------|
| Feature describing pose of the gripper and the grasped object relative to the target object.| Placing a grasped object on another object |
| feature describing pose of the gripper relative to target.| Reaching for a specific object |
| feature describing pose of the cup relative to cup and fork.| Placing a grasped cup next to plate and fork |

* Question : How do we map between this feature and action ? How to generate the action from 1 feature ? Is this action related to the action of PDDL ?

## Expert 'e' :
 * Please can you explain experts with respect to the given examples.
     * How many experts were used ?
     * what values did they suggest ?

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
* Do you use PDDL ?
* Do you use DMP for trajectory generation ?
* Once the relavant feature is selected what software did you use to generate
the trajectory ?


