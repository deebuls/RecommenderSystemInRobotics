MIR Teleop Record
==========

The node is used to record the readings of all the frames of the robot and 
to store it in a json file.

The arms are controlled using the logistic joypad. Using the Joypad the arms 
are moved to the respective initial and final positions. After reaching the 
respective positions the record buttions are pressed and the recordings of 
the Joint angles, Joint positions are recorded in a json file. The json file 
can has the current time stamp in its header.

## Recording Keys

![mir_teleop_record/ros/doc/youbot_joypad_description.pdf]
Please check the above file for key bindings.

### Recording Initial Value to File
Key(Print arm Joint States) + Key (Arm Joint 1 and 2)
### Recording Final Value to File
Key(Print arm Joint States) + Key (Arm Joint 2 and 3)

The frames used can be obtained from the following command 
> rosrun tf tf_monitor

![frames ](frames.png)

The frames can be modified as per changes in the file 
teleop_joypad_record_node.cpp .


```cpp
const std::string TeleOpJoypad::frame_name_[9] =
    {"arm_link_0", "arm_link_1", "arm_link_2", "arm_link_3", "arm_link_4", 
     "arm_link_5", "gripper_palm_link", "gripper_finger_link_l","gripper_finger_link_r" };
```


Update the frame string list to add or delete required frames.
