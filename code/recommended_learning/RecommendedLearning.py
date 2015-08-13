#%matplotlib inline
import glob
import json
import math #for log base 2 
import pandas.io as io
import pandas as pd
import numpy as np
import quaternion
import sklearn.preprocessing as pre
import pprint
import matplotlib.pyplot as plt
from collections import defaultdict

##Library based on statsmodel for information theory
import infotheo

class DataAnalyzer:
    def __init__(self, folderPath, skill, bins=9):
        self.skill_being_learned = skill
        self.initial = io.json.read_json( folderPath + "/INITIAL.json", typ='frame')
        self.final = io.json.read_json( folderPath + "/FINAL.json", typ='frame')
        
        self.final = self.final.reindex(np.random.permutation(self.final.index))
        #Number of bins in the pdf function creation relates to the precision of the 
        self.num_of_bins = bins
        

        #Deleting state column as not reqired for calculations
        del self.initial['state']
        del self.final['state']
        
        print "Are the inital and final features equal : ", (self.initial.columns.values==self.final.columns.values).all()
        
        self.readExpertKnowledgeBase()
        number_of_demonstrations = self.initial['arm_joint_1'].count()
        print self.initial['arm_joint_5']
        print("Data has been succesfully read. Nu of Demonstrations : ", number_of_demonstrations)
        
                   
        self.nod = number_of_demonstrations
        self.check_data_for_nan()
            
        
    def check_data_for_nan(self):
        """
        Data collection on the hardware showed missing some values 
        Filing those data with average.
        Need to fix this while taking demonstration.
        """
        null_in_data = self.initial.isnull().sum()
        #print "Number of null initial : ", null_in_data[null_in_data != 0]
        null_in_data = self.final.isnull().sum()
        #print "Number of null final : ", null_in_data[null_in_data != 0]
        if (self.initial.isnull().any().any()):
            print "Initial has null values, Filling with average : "
            self.initial = self.initial.fillna(self.initial.mean())
        if (self.final.isnull().any().any()):
            print "Final has null values, Filling with average : "
            self.final = self.final.fillna(self.final.mean())
        
            
    def mad_based_outlier(self, points, thresh=3.5):
        """
        http://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
        Removes outliers based on MAD.
        Returns boolean map with false value in outliers
        """
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh
        
    def readExpertKnowledgeBase(self):
        '''
        The function reads the knowledge base to extract the mappings between the 
        skill to features.
        
        ---------------------------------------------------------------
        |skill \ Features || Feature1 | Feature2 | Feature3 | Feature4 |
        ---------------------------------------------------------------
        |skill1           ||    *     |          |          |          |
        |skill2           ||          |     *    |     *    |          |
        |skill3           ||          |          |          |     *    |
        |skill4           ||     *    |          |     *    |     *    |
        ---------------------------------------------------------------
        
        TODO :
        how to represent the knoweldge base
        how to intutively add the knowledge base
        
        '''
        #Database Path
        self.skills_db = './knowledge_base/skills.csv'
        self.robot_features_db = './knowledge_base/robot_features.csv'
        self.environment_features_db = './knowledge_base/environment_features.csv'
        self.templates_db = './knowledge_base/templates.csv'
        
        #Creating knowledge base
        self.skills = []
        self.robot_features = []
        self.environment_features = []
        templates = []
        self.templates = defaultdict(list)
        self.features_which_have_changed = [] #is updated in the entropy function
        
        #Reading knowledge Base
        with open(self.skills_db, 'rb') as f:
            for line in f:
                self.skills.append(line.rstrip('\n'))
        with open(self.robot_features_db, 'rb') as f:
            for line in f:
                self.robot_features.append(line.rstrip('\n'))
        with open(self.environment_features_db, 'rb') as f:
            for line in f:
                self.environment_features.append(line.rstrip('\n'))
        with open(self.templates_db, 'rb') as f:
            for i, line in enumerate(f):
                templates.append(line.rstrip('\n').split())
                for temp in line.rstrip('\n').split():
                    self.templates[i].append(temp)
                    
        self.features_without_positions = {'arm_joint_1','arm_joint_2','arm_joint_3','arm_joint_4','arm_joint_5'} 
        self.combined_features_from_readings = self.initial.columns.values
        self.manipulate_templates()
        
                
        print "Database reading Complete:"
        print "Skills : ",len(self.skills)," Robot Features : ",len(self.robot_features)
        print "Environment Features : ",len(self.environment_features)," Templates : ", len(self.templates)
            
    def manipulate_templates(self):
        '''
        The expert only specifies the feature name in the template
        Internal replresentation conatins splitting data in x y z and orientation
        This funciton splits the template feature name into internal representation
        
        This function also adds the distance feature of all the features with the environment features.
        
        '''
        
        for i, temp in self.templates.iteritems():
            template_robot_features, template_env_features = self.split_template(temp)
            if temp[2] == 'absolute' :   #checking if the 2nd variable in template is absolute or  relative
                for feature in temp[4:]:           #without considering the template name and the type provided
                    if feature not in self.features_without_positions:
                        #check if the features values are provided as split or standalone 
                        # if standalone split the various features positions and orientation
                        if feature not in self.combined_features_from_readings:
                            self.templates[i].remove(feature)
                            if temp[3] == 'position' or temp[3] == 'both':
                                self.templates[i].extend([feature + "_x", feature + "_y", feature + "_z"])

                            if temp[3] == 'orientation' or temp[3] == 'both':
                                self.templates[i].extend([feature + "_ox",feature + "_oy",feature + "_oz", feature + "_ow"])
            elif temp[2] == 'relative' :
                
                for t_robot_feature in template_robot_features:
                        
                    for t_env_feature in template_env_features:
                        '''
                        try:
                            #self.templates[i].remove(t_robot_feature)
                            except:
                            pass
                        '''
                        try:
                            self.templates[i].remove(t_env_feature)
                            self.templates[i].remove(t_robot_feature)
                        except:
                            pass
                        if temp[3] == 'position' or temp[3] == 'both':
                            self.templates[i].append("d_linear_"+t_robot_feature+"_"+t_env_feature)
                        if temp[3] == 'orientation' or temp[3] == 'both':
                            self.templates[i].append("d_angular_"+t_robot_feature+"_"+t_env_feature)


    def split_template(self, temp):
        '''
        Splits template into robot feature and env feature 
        '''
        template_robot_features = []
        template_env_features = []
        for feature in temp:
            if feature in self.robot_features and feature not in self.features_without_positions:
                template_robot_features.append(feature)
            if feature in self.environment_features:
                template_env_features.append(feature)
                
        return template_robot_features, template_env_features
    
    def check_if_robot_feature_changed(self, feature):
        """
        For relative templates 
        the distance between changes even when the robot didnt move.
        So nned to check if the robot parameters had changed to consider the 
        distance.
        """
        features = [feature + "_x", feature + "_y", feature + "_z",
                    feature + "_ox",feature + "_oy",feature + "_oz", feature + "_ow"]
        for feature in features:
            if feature in self.features_which_have_changed:
                return True
        #No Parameter has changed return False
        return False
        
        
                
    def recommend_using_knowledge_base(self):
        """
        After Calculation of the entropy.
        The Templates for each action are used to determine the feature 
        with lowest entropy
        """

        print "skill being learnt : ", self.skill_being_learned
        
        for key, temp in self.templates.iteritems():
            entropy_sum_template = 0
            cond_entropy_sum_template = 0
            if self.skill_being_learned == temp[0]:
                num_of_features = len(temp[4:]) #1 skill type, 2 effect metric, 3 pos/orin , from 4 features
                print "Calc entropy for Template :",key,temp[0:3],
                for feature in temp[4:]: 
                    #Relative Template the robot skill is not removed for checking for movement
                    if feature in self.robot_features and feature not in self.features_without_positions:
                        #checks if the robot moved and sets flag accordingly
                        robot_feature_changed_flag = self.check_if_robot_feature_changed(feature)
                        #print 'For Relative Template, Robot Feature change flag : ', robot_feature_changed_flag
                        #if the robot didnt move then the distance parmeters are not added but penality are added 
                        if not robot_feature_changed_flag:
                            entropy_sum_template += 1
                            cond_entropy_sum_template += 1
                        continue

                    if (feature in self.features_which_have_changed ) :  #checking if the feature was changed in the demonstration
                        try:
                            print feature,
                            entropy_sum_template += self.shannonEntropyFinal[feature]
                            cond_entropy_sum_template += self.condEntropyFinalGivenInitialDict[feature]
                        except :
                            print "feature not available : ", feature
                            pass
                    else :  #if feature was not changed giving higher value
                        print "(",feature,")",
                        entropy_sum_template += 1
                        #Templates whose all value are modifie should be selected than partial modified templates
                        #As some large templates with just 1 matching takes over eligible ones
                        cond_entropy_sum_template += 1
                        #reducing the number of features utilized 
                        #num_of_features = num_of_features -1
                        
                print "entropy : ",entropy_sum_template, " features : ", num_of_features
                        
            self.templates[key].insert(4, entropy_sum_template)
            self.templates[key].insert(5, num_of_features)
            self.templates[key].insert(6, cond_entropy_sum_template)
        
        templates_df = pd.DataFrame(temp for temp in self.templates.values())   
        self.recommended_templates = templates_df.sort([4,5], ascending= [1,0]).iloc[:,:6]
        cols = ['skill', 'section', 'metrics', 'geometry', 'entropy', 'num of features']
        self.recommended_templates.columns = cols
        print self.recommended_templates
        for i in range(3):
            print self.recommended_templates.iloc[i].name,
        print
        self.recommended_templates_cond = templates_df.sort([6,5], ascending= [1,0]).iloc[:,:7]
        cols = ['skill', 'section', 'metrics', 'geometry', 'entropy', 'num of features', 'cond entropy']
        self.recommended_templates_cond.columns = cols
        print self.recommended_templates_cond
        
    def dataManipulation(self):
        '''
        TODO : Remove this once the camera based object detector is ready
        '''
        #Assigning griper palm co - ordinates as the object point
        self.final = self.final.assign(object_1_x = self.final.gripper_palm_link_x, 
                     object_1_y = self.final.gripper_palm_link_y,
                     object_1_z = self.final.gripper_palm_link_z,
                     object_1_ox = self.final.gripper_palm_link_ox,
                     object_1_oy = self.final.gripper_palm_link_oy,
                     object_1_oz = self.final.gripper_palm_link_oz,
                     object_1_ow = self.final.gripper_palm_link_ow)

        self.initial = self.initial.assign(object_1_x = self.final.gripper_palm_link_x, 
                             object_1_y = self.final.gripper_palm_link_y,
                             object_1_z = self.final.gripper_palm_link_z,
                             object_1_ox = self.final.gripper_palm_link_oy, #mixing the pose to get different pose between initial and final
                             object_1_oy = self.final.gripper_palm_link_ox,
                             object_1_oz = self.final.gripper_palm_link_ow,
                             object_1_ow = self.final.gripper_palm_link_oz)

                
        
        
    def dataCalculatingRelativeDistances(self, from_frame, to_frame):
        '''
        This function calculates all the relative distance between all the frames .
        Both linear distance and angular distance.
        '''
        linear_distance = "d_linear_"+from_frame+"_"+to_frame
        angular_distance = "d_angular_"+from_frame+"_"+to_frame
        from_x = from_frame+"_x"
        from_y = from_frame+"_y"
        from_z = from_frame+"_z"
        to_x = to_frame+"_x"
        to_y = to_frame+"_y"
        to_z = to_frame+"_z"
        from_ox = from_frame+"_ox"
        from_oy = from_frame+"_oy"
        from_oz = from_frame+"_oz"
        from_ow = from_frame+"_ow"
        to_ox = to_frame+"_ox"
        to_oy = to_frame+"_oy"
        to_oz = to_frame+"_oz"
        to_ow = to_frame+"_ow"
        
        '''
        self.final = self.final.assign(d_linear_m0_gripper = lambda x: np.sqrt((x.gripper_palm_link_x - x.object_1_x)**2 +
                                                              (x.gripper_palm_link_y - x.object_1_y)**2 +
                                                              (x.gripper_palm_link_z - x.object_1_z)**2 ))
        
        self.initial = self.initial.assign(d_linear_m0_gripper = lambda x: np.sqrt((x.gripper_palm_link_x - x.object_1_x)**2 +
                                                                      (x.gripper_palm_link_y - x.object_1_y)**2 +
                                                                      (x.gripper_palm_link_z - x.object_1_z)**2 ))
        '''
        try :
            # Calculating Distance between gripper and object and assigning 
            final_distance_col = np.sqrt((self.final[from_x] - self.final[to_x])**2 +
                                         (self.final[from_y] - self.final[to_y])**2 +
                                         (self.final[from_z] - self.final[to_z])**2 )
            init_distance_col = np.sqrt((self.initial[from_x] - self.initial[to_x])**2 +
                                        (self.initial[from_y] - self.initial[to_y])**2 +
                                        (self.initial[from_z] - self.initial[to_z])**2 )

            self.final[linear_distance] = final_distance_col
            self.initial[linear_distance] = init_distance_col
            #self.final = self.final.assign(linear_distance = final_distance_col)
            #self.initial = self.initial.assign(linear_distance = init_distance_col)

            # Calculating angular distances between the frames 
            diffAngle = []

            for index, row in self.final.iterrows():
                q0 = np.quaternion(row[to_ox], row[to_oy], row[to_oz], row[to_ow])
                q1 = np.quaternion(row[from_ox], row[from_oy], row[from_oz], row[from_ow])
                _ = q0.inverse()*q1
                diffAngle.append(_.angle())

            self.final[angular_distance] = diffAngle

            diffAngle = []
            for index, row in self.initial.iterrows():
                q0 = np.quaternion(row[to_ox], row[to_oy], row[to_oz], row[to_ow])
                q1 = np.quaternion(row[from_ox], row[from_oy], row[from_oz], row[from_ow])
                _ = q0.inverse()*q1
                diffAngle.append(_.angle())

            self.initial[angular_distance] = diffAngle

        except :
            pass
            #print "Relative Distance : Feature not present in Readings : ", from_frame," ",to_frame
        
    def createRelativeDistanceData(self):
        '''
        Calculates the relative distance between the features which are relevant
        This is a general function to keep adding the features between which the 
        data has to be calculated.
        The the distance between Robot features and environment features.
        Features :
        Robot Features :
        "arm_link_0", "arm_link_1", "arm_link_2", "arm_link_3", "arm_link_4",
        "arm_link_5", "gripper_palm_link", "gripper_finger_link_l","gripper_finger_link_r",
        "base_footprint", "base_link", "wheel_link_bl", "wheel_link_br", "wheel_link_fl", "wheel_link_fr" 
        
        Environment Features :
        "table_1","table_2","table_3","table_4","table_5","table_6","table_7","object_1"
        '''
        
        #TODO the base frame for TF transformation was taken as arm_link_1 so its missing in the 
        #data collection. So need to update .
        robot_features = ["arm_link_0", "arm_link_2", "arm_link_3", "arm_link_4",
        "arm_link_5", "gripper_palm_link", "gripper_finger_link_l","gripper_finger_link_r",
        "base_footprint", "base_link", "wheel_link_bl", "wheel_link_br", "wheel_link_fl", "wheel_link_fr" ]
        
        env_features = ["table_1","table_2","table_3","table_4","table_5","table_6","table_7"]
        print "env : ",self.environment_features

        for robot_feature in self.robot_features:
            if robot_feature not in self.features_without_positions:
                for env_feature in self.environment_features:
                    self.dataCalculatingRelativeDistances(robot_feature, env_feature)
                
        print("Relative Distances are calculated . Data is ready for analysis")
        print "Total number of features : ",len(self.initial.columns.values)

        
        
  
        
    def discribeParameter(self, name, number_of_demonstrations=None):
         #if None select all demonstrations
        if number_of_demonstrations == None or number_of_demonstrations > self.nod:
            num = self.nod
        else:
            num = number_of_demonstrations
        print "Parameter :", name 
        print "INITIAL :", self.initial[name]
        print "FINAL :", self.final[name]
        #plt.savefig("/data/dataDeebul/rnd/RecommenderSystemInRobotics/experiments/" + name + "Box")

        pdf, H, xedges, yedges = self.jointProbabilityDensityFunction(name, self.num_of_bins, num)
        
        px = pdf.sum(0)
        py = pdf.sum(1)
        print pdf 
        print "pdf Initial Values, pdf Final Values  :", py, px
        print "check sumpx, sum py, sumpxpy : ", sum(px), sum(py), sum(px)+sum(py)
        print "H_FinalGivenInitial : ",  np.float16(infotheo.condentropy(px, py, pdf)),
        print "H_InitialGivenFinal :", np.float16(infotheo.condentropy(py, px, pdf))
        
        initialValue = np.asarray( self.initial[name][:num] )
        finalValue = np.asarray( self.final[name][:num])
        
        if (abs(np.mean(initialValue) - np.mean(finalValue) )> 0.1 )  :
            if (np.std(initialValue) > np.std(finalValue)):
                print "Changed Feature and converged : ", name
            else:
                print "Feature changed but not converged "
            
        else :
            print " Feature not changed : ", np.mean(initialValue), np.mean(finalValue) 
        
        
        labels = list('IF')
        plt.boxplot(np.vstack((initialValue,finalValue)).T, labels=labels)
        plt.show()
        #min_max_scaler = preprocessing.MinMaxScaler()
        #initialValue = min_max_scaler.fit_transform(initialValue)
        #finalValue = min_max_scaler.fit_transform(finalValue)
        
        initialValue = np.around(initialValue, decimals=4)
        finalValue = np.around(finalValue, decimals=4)
        
        
        #Return a boolean map of data, with false at place of all outliers
        #So replacing all the outliers with the std mean 
        initialValue[self.mad_based_outlier(initialValue)] = np.median(initialValue, axis=0)
        finalValue[self.mad_based_outlier(finalValue)] = np.median(finalValue, axis=0)
  
  
        myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
        plt.imshow(H,origin='low',extent=myextent,interpolation='nearest',aspect='auto')
        plt.plot(finalValue, initialValue,'ro')
        plt.colorbar()
        plt.ylabel("Initial Values")
        plt.xlabel("Final Values")
        plt.title("Parameter : "+name )
        #plt.savefig("/data/dataDeebul/rnd/RecommenderSystemInRobotics/experiments/" + name + "JoinPDF")
        plt.show()
        
        
        
    def jointProbabilityDensityFunction(self, feature, bins=10, number_of_demonstrations=None):
        """
        Creates the Joint probability distribution based on the initial
        and final values of the feature.

        """
        #if None select all demonstrations
        if number_of_demonstrations == None :
            num = self.nod
        else:
            num = number_of_demonstrations
            
        
        initialValue = np.asarray( self.initial[feature][:num] )
        finalValue = np.asarray( self.final[feature][:num])
        if number_of_demonstrations < 3 :
            for i in range(3 - number_of_demonstrations ):
                initialValue = np.append(initialValue, initialValue[0])
                finalValue = np.append(finalValue, finalValue[0])
        
        #min_max_scaler = preprocessing.MinMaxScaler()
        #initialValue = min_max_scaler.fit_transform(initialValue)
        #finalValue = min_max_scaler.fit_transform(finalValue)
        
        initialValue = np.around(initialValue, decimals=3)
        finalValue = np.around(finalValue, decimals=3)
        
        #Return a boolean map of data, with false at place of all outliers
        #So replacing all the outliers with the std mean 
        initialValue[self.mad_based_outlier(initialValue)] = np.median(initialValue, axis=0)
        finalValue[self.mad_based_outlier(finalValue)] = np.median(finalValue, axis=0)
        #pre.normalize((initialValue,finalValue),copy=False)

        #scalling both the axes on the same scale 
        #since intital and final values are measured of a single feature
        #The histogram should be on same scale 
        #from larget value of both to smallest value of both
        value_max = max(initialValue.max(), finalValue.max())
        value_min = min(initialValue.min(), finalValue.min())
        range_value = [[value_min, value_max],[value_min, value_max]]
        threshold = abs(value_max - value_min)/bins
        change_in_mean = abs(np.mean(initialValue) - np.mean(finalValue) )
        #print feature, threshold, change_in_mean, (change_in_mean > threshold ), change_in_mean > 0.1
        
        #checking if the mean has changed to identify the features which have changed
        if (change_in_mean > threshold )  :
            #print "Changed Feature : ", feature
            #checking for convergence in deviation to identify converged values
            if (np.std(initialValue) > np.std(finalValue)):
                self.features_which_have_changed.append(feature)
        
        
        H, xedges, yedges = np.histogram2d(finalValue, initialValue, bins=bins, range=range_value)
        return (H/float(len(initialValue))).T, H.T, xedges, yedges 

    def condEntropyFinalGivenInitial(self, number_of_demonstrations = None):
        """
        Calculates the conditional entropy of the features. It calculates the 
        final entropy given initial 
        """
            
        self.condEntropyFinalGivenInitialDict = {}
        self.condEntropyInitialGivenFinalDict = {}
        self.shannonEntropyFinal = {}
        self.features_with_zero_entropy = list()

        for featureName in self.initial.columns.values:
            pdf, _, __, ___ = self.jointProbabilityDensityFunction(featureName, self.num_of_bins, number_of_demonstrations)
            #pdf of initial values
            pFinal = pdf.sum(0)
            #pdf of final values
            pInitial = pdf.sum(1)
            
            try:
                #Entropy of Final given Initial
                self.condEntropyFinalGivenInitialDict[featureName] = np.float16(infotheo.condentropy( pFinal, pInitial, pdf))
                self.condEntropyInitialGivenFinalDict[featureName] = np.float16(infotheo.condentropy( pInitial, pFinal, pdf))
                self.shannonEntropyFinal[featureName] = np.float16(infotheo.shannonentropy(pFinal))
            except:
                print "[ERROR ] Feature the probability is wrong : ", featureName
            #print "Entropy : ",featureName," : ",self.condEntropyFinalGivenInitialDict[featureName]," : ", np.float16(infotheo.condentropy( pInitial, pFinal, pdf))
         
        print "Number of Features which have changed : ", len(self.features_which_have_changed)
        #print "Zero Entropy Features : ", sum( (x==0.0 or np.isinf(x)) and self.condEntropyInitialGivenFinalDict[name] != 0.0 for name,x in self.condEntropyFinalGivenInitialDict.iteritems() )
        
        '''
        for name,FgivenI in self.condEntropyFinalGivenInitialDict.iteritems():
            IgivenF = self.condEntropyInitialGivenFinalDict[name]
            if (FgivenI == 0.0 or np.isinf(FgivenI)) and  (IgivenF != 0.0 or not np.isinf(IgivenF)) :
                self.condEntropyFinalGivenInitialDict[name] = 0.0
                self.features_with_zero_entropy.append(name)
        '''
        for name, e in self.shannonEntropyFinal.iteritems():
            if e == 0.0:
                self.features_with_zero_entropy.append(name)
        print "zero entropy features : ", len(set(self.features_with_zero_entropy))       
        print "Features that have changed and zero entropy : ", len(set(self.features_which_have_changed) & set(self.features_with_zero_entropy))
        #print set(self.features_which_have_changed) & set(self.features_with_zero_entropy)
        return(self.condEntropyFinalGivenInitialDict)

    def condEntropyInitialGivenFinal(self, number_of_demonstrations = None):
        """
        Calculates the conditional entropy of the features. It calculates the 
        final entropy given initial 
        """
        
        
        if (self.initial.columns.values==self.final.columns.values).all() == False :
            print ("Columns are not same. Error in data collection")
        
        
        self.condEntropyInitialGivenFinal = {}
        for featureName in self.initial.columns.values:
            pdf, _, __, ___ = self.jointProbabilityDensityFunction(featureName, self.num_of_bins, number_of_demonstrations )
            #pdf of initial values
            pInitial = pdf.sum(0)
            #pdf of final values
            pFinal = pdf.sum(1)
            
            #Entropy of Final given Initial
            self.condEntropyInitialGivenFinal[featureName] = np.float16(infotheo.condentropy( pInitial, pFinal, pdf))
            

        return(self.condEntropyInitialGivenFinal)
    
    def predictGoalValue(self):
        selected_template = self.recommended_templates.iloc[0]
        if selected_template['section'] == 'base' :
            template_num =  selected_template.name
            features = self.templates[template_num]
            for feature in features[4:]:
                Value = np.asarray( self.final[feature])

                print feature

