from pykalman import KalmanFilter
import pandas as pd
import numpy as np 
from cv_service.topics import cv_namespace
from cv_service.components.cv_component import CVComponent

class InterpolatorComponent(CVComponent):
    INPUT_TOPICS = ['cv_service.topics.cv_namespace.ObjectDescriptor']
    OUTPUT_TOPICS = ['cv_service.topics.cv_namespace.Track']

    def __init__(self, *args, **kwargs):
        super(InterpolatorComponent, self).__init__(*args, **kwargs)
        # initialize kalman filter parameters
        self.F = [1]  # state transition equation
        self.Q = 0.1  # transition covariance        
        self.H = [1]  # observation matrix
        self.R = [2] # observation covariance  ****ADJUST BASED ON SENSOR NOISE**** n = unit of measurement (e.g. 1 = 1 meter)
        self.initial_state_mean = 0
        self.P = [0.001]   # initial state covariance.  large values if guessing initial positions
        
        self._message_df = pd.DataFrame(columns=['timestamp', 'object_id', 'age_error', 'age_value', 
                    'gender_label', 'gender_score', 'position_x', 'position_y'])
        self._result = 0

    def receive(self, topic, message):
        # convert message class object to dict
        message = vars(message).get('_properties')
        message['timestamp'] = ((message['timestamp'] + 50)/100)*100
        # establish object ID, X, Y measurements for Kalman Filter
        self._message_df = self._message_df.append(message, ignore_index=True)        
        self._message_df = self._message_df.drop_duplicates(subset=['timestamp', 'object_id'], keep='last')

        def next(obj, field):
            objvals = self._message_df[self._message_df['object_id'] == obj][field]
            if len(self._message_df[self._message_df['object_id'] == obj]) < 2:
                self._result = message.get(field, None)
            else:
                self.initial_state_mean = objvals.iat[0]
                self.kf = KalmanFilter(
                    transition_matrices=self.F, 
                    transition_covariance=self.Q, 
                    observation_matrices=self.H, 
                    observation_covariance=self.R,
                    initial_state_mean=self.initial_state_mean, 
                    initial_state_covariance=self.P)        
                state_means, _ = self.kf.filter(objvals.values)
                state_means = pd.Series(state_means.flatten())
                self._result = state_means.iloc[-1]
                if len(objvals) < 10:
                    pass
                else:
                    self._message_df[self._message_df['object_id'] == obj].drop(
                    self._message_df[self._message_df['object_id'] == obj].head(1).index)
            return self._result


        # convert dictionary key values to values
        ts = message.get('timestamp', None)
        id = message.get('object_id', None)
        x_loc = next(id, 'position_x')
        y_loc = next(id, 'position_y')

        new_message = cv_namespace.Track(
            timestamp=ts,
            object_id=id,
            age_error=message.get('age_error', None),
            age_value=message.get('age_value', None),
            gender_label=message.get('gender_label', None),
            gender_score=message.get('gender_score', None),
            positions=[
                cv_namespace.TrackPosition(
                    timestamp=ts,
                    x=x_loc,
                    y=y_loc
                ),
            ]
        )
        self.publish('cv_service.topics.cv_namespace.Track', new_message)
