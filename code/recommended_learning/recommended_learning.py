from formlayout import fedit
from StringIO import StringIO

class recommended_learning:
    def __init__(self):
        #initalize filenames
        self.skills_db = './knowledge_base/skills.csv'
        self.robot_features_db = './knowledge_base/robot_features.csv'
        self.environment_features_db = './knowledge_base/environment_features.csv'
        self.templates_db = './knowledge_base/templates.csv'
        self.read_database()
        datagroup = self.create_initial_datagroup()
        result = fedit(datagroup, title="Recommended Learning YouBot ")
        print result
        self.edit_database(result[1])
        self.edit_templates(result[3])

    def edit_database(self, data):
        if (data[0] != ''):
            with open(self.skills_db, 'ab') as f:
                print "Adding to skill database : ",data[0]
                f.write(data[0])
                f.close()
        if (data[1] != ''):
            if (data[2] == 0 ):
                print "Adding to Robot Feature database : ",data[1]
                with open(self.robot_features_db, 'ab') as f:
                    f.write(data[1])
                    f.close()
            if (data[2] == 1 ):
                print "Adding to Environemnt Feature database : ",data[1]
                with open(self.environment_features_db, 'ab') as f:
                    f.write(data[1])
                    f.close()

    def edit_templates(self, data):
        print data
        with open(self.templates_db, 'ab') as f:
            f.write(self.skills[data[0]])
            f.close()

        with open(self.templates_db, 'a') as f:
            for d in data[1:]:
                if d != 0:
                    f.write(" " + self.robot_features[d])
            f.write("\n")
            f.close()


    def read_database(self):
        self.skills = []
        self.robot_features = []
        self.environment_features = []
        self.templates = []
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
            for line in f:
                self.templates.append(line.rstrip('\n'))

        #appending none in database initial
        self.robot_features = ['none']+self.robot_features
        self.environment_features = ['none']+self.environment_features


    def create_initial_datagroup(self):
        learn_list = self.create_learning_datalist()
        edit_list = self.create_edit_datalist()
        show_list = self.create_show_datalist()
        edit_template_list = self.create_edit_template_datalist()
        return ((learn_list, "Learn", "Start Learning "),
                (edit_list, "Edit","Edit Skill or features" ),
                (show_list, "Show", "Display all the skill "),
                (edit_template_list, "Add Template", "Add Template for each skills"))

    def create_learning_datalist(self):
        return [('Start Learning', False),
                ('Skill',  [0]+[(i, s) for i, s in enumerate(self.skills)]),
                ('Name', '')]

    def create_edit_template_datalist(self):
        return [('Skill', [0]+[(i, s) for i, s in enumerate(self.skills)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)]),
                ('Robot Feature',[0]+[(i,f) for i , f in enumerate(self.robot_features)])]

    def create_edit_datalist(self):
        return [('Add Skill', ''),
                ('Add Feature', ''),
                ('Type of Feature', [0, "Robot Feature", "Environment Feature"])]

    def create_show_datalist(self):
        return [('Skills', '\n'.join(self.skills)),
                ('Robot Features', '\n'.join(self.robot_features)),
                ('Environment  Features', '\n'.join(self.environment_features)),
                ('Templates', '\n'.join(self.templates))]

if __name__ == "__main__":
    recommended_learning()
