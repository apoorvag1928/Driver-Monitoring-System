
#import packages
import pyforms
from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlButton
from driver_monitoring_system import driver_monitoring_system
from drowsiness_detection import drowsiness_detection
from talking_detection import talking_detection
from head_pose_detection import head_pose_detection

#defining user interface class
class user_interface(BaseWidget):
    
    #main function
    def __init__(self):

        #giving window name
        super(user_interface, self).__init__('WELCOME: CHOOSE YOUR MODE')
        
        #defining css for styling
        self.setStyleSheet("""
        QWidget { background: lightgrey; padding: 5px 10px 5px 10px; }
        QPushButton { background: #1C8DEB; color: white; padding: 10px 20px 10px 20px; border-radius: 5px; font-weight: bold; font-size: 20px; font-style: times new roman}
        QPushButton:hover { background: #3080CC; color: black; }
        """)      

        #defining buttons for different modes
        self.start_drowsiness = ControlButton('Start Drowsiness Detection')
        self.start_drowsiness.value = self.button_action_1
        
        self.start_talking= ControlButton('Start Talking Detection')
        self.start_talking.value = self.button_action_2

        self.start_head = ControlButton('Start Head Pose Detection')
        self.start_head.value = self.button_action_3
        
        self.start_driver = ControlButton('Start Driver Monitoring System')
        self.start_driver.value = self.button_action_4


    #on click call drowsiness detection class
    def button_action_1(self):
        drowsy = drowsiness_detection()        


    #on click call talking detection class
    def button_action_2(self):
        talk = talking_detection()

    #on click call head pose detection class
    def button_action_3(self):
        head_pose = head_pose_detection()

    #on click call driver monitoring system class
    def button_action_4(self):
        driver = driver_monitoring_system()
        
    
#Execute the application
if __name__ == "__main__":   
    pyforms.start_app(user_interface, geometry=(500, 215, 320, 215))

