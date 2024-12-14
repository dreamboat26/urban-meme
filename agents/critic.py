from keras import layers, optimizers, models
from keras import backend as K

class critic():
    def __init__(self, state_size, action_size, h1, h2, lr):
        self.n_states = state_size
        self.n_action = action_size
        self.h_1 = h1
        self.h_2 = h2
        self.critic_lr = lr
        
        self.build_model()
        
    def build_model(self):
        input_states = layers.Input(shape=(self.n_states,), name='states')
        input_actions = layers.Input(shape=(self.n_action,), name='actions')
        
        # State Network
        state_h1 = layers.Dense(units=self.h_1, use_bias=False)(input_states)
        state_h1 = layers.BatchNormalization()(state_h1)
        state_h1 = layers.Activation(activation='relu')(state_h1)
        state_h1 = layers.Dropout(0.2)(state_h1)
        
        state_h2 = layers.Dense(units=self.h_1, use_bias=False)(state_h1)
        state_h2 = layers.BatchNormalization()(state_h2)
        state_h2 = layers.Activation(activation='relu')(state_h1)
        state_h2 = layers.Dropout(0.3)(state_h2)
        
        # Action Network
        action_h1 = layers.Dense(units=self.h_1, use_bias=False)(input_actions)
        action_h1 = layers.BatchNormalization()(action_h1)
        action_h1 = layers.Activation(activation='relu')(action_h1)
        action_h1 = layers.Dropout(0.4)(action_h1)
        
        action_h2 = layers.Dense(units=self.h_1, use_bias=False)(action_h1)
        action_h2 = layers.BatchNormalization()(action_h2)
        action_h2 = layers.Activation(activation='relu')(action_h2)
        action_h2 = layers.Dropout(0.4)(action_h2)
        
        # Merge two layers
        critic_h2 = layers.Add()([state_h2, action_h2])
        critic_h2 = layers.Activation(activation='relu')(critic_h2)
        
        critic_h3 = layers.Dense(units=self.h_2, use_bias=False)(critic_h2)
        critic_h3 = layers.BatchNormalization()(critic_h3)
        critic_h3 = layers.Activation(activation='relu')(critic_h3)
        critic_h3 = layers.Dropout(0.3)(critic_h3)
        
        Q_value = layers.Dense(units=1, name='q_value')(critic_h3)
        
        self.criticModel = models.Model(inputs=[input_states, input_actions], outputs=Q_value)
        
        optimizer = optimizers.Adam()
        self.criticModel.compile(optimizer, loss='mse')
        
        # Compute gradient of Q_value w.r.t. input action values
        action_gradients = K.gradients(Q_value, input_actions)
        
        # to deliever action_gradients to actor while calling this method from actor
        self.get_action_gradients = K.function(inputs=[*self.criticModel.input, K.learning_phase()], 
                                                       outputs=action_gradients)