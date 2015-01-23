# Other Applications of Recommender Systems
=============================

* Preferences of users in Task Planing
* Kernel Recommendation in SVM
    * Firstly, linearity is rather special, and outside quantum mechanics no
      model of a real system is truly linear. Secondly, detecting linear
relations has been the focus of much research in statistics and machine
learning for decades and the resulting algorithms are well understood, well
developed and efficient. Naturally, one wants the best of both worlds. So, if a
problem is non-linear, instead of trying to fit a non-linear model, one can map
the problem from the input space to a new (higher-dimensional) space (called
the feature space) by doing a non-linear transformation using suitably chosen
basis functions and then use a linear model in the feature space. This is known
as the ‘kernel trick’. The linear model in the feature space corresponds to a
non-linear model in the input space. This approach can be used in both
classification and regression problems. The choice of kernel function is
crucial for the success of all kernel algorithms because the kernel constitutes
prior knowledge that is available about a task. Accordingly, there is no free
lunch (see No Free Lunch Theorems) in kernel choice.[1]
    * Recommender system : Collabrative Filter based can be used to determine 
which algo to use.

* Motion/Path Planning algo in Planning 
    * OMPL has implemented the following planners
      http://ompl.kavrakilab.org/planners.html
    * Which planner to select can be done automatically explained here 
        "How OMPL selects a control-based planner
If you use the ompl::control::SimpleSetup class (highly recommended) to define
and solve your motion planning problem, then OMPL will automatically select an
appropriate planner (unless you have explicitly specified one). If the state
space has a default projection (which is going to be the case if you use any of
the built-in state spaces), then it will use KPIECE. This planner has been
shown to work well consistently across many real-world motion planning
problems, which is why it is the default choice. In case the state space has no
default projection, RRT will be used. Note that there are no bidirectional
control-based planners, since we do not assume that there is a steering
function that can connect two states exactly."
    * Again Recommender Collabrative filter can be used to decide which algo to
      be used for recommendation.

1. http://www.svms.org/kernels/
