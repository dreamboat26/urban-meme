from keras import layers, models, optimizers
from keras import backend as K

class actor():
    def __init__(self, state_size, action_size, h1, h2, lr, r_h, r_l):
        self.n_state = state_size
        self.n_action = action_size
        self.h_1 = h1
        self.h_2 = h2
        self.lr_ = lr
        self.rotor_high = r_h
        self.rotor_low = r_l
        
        self.build_model()
        
    def build_model(self):
        # Define Input Layer
        input_states = layers.Input(shape=(self.n_state,), name="input")
        
        h_1 = layers.Dense(units=self.h_1, use_bias=False)(input_states)
        h_1 = layers.BatchNormalization()(h_1)
        h_1 = layers.Activation(activation='relu')(h_1)
        h_1 = layers.Dropout(0.3)(h_1)
        
        h_2 = layers.Dense(units=self.h_2, use_bias=False)(h_1)
        h_2 = layers.BatchNormalization()(h_2)
        h_2 = layers.Activation(activation='relu')(h_2)
        h_2 = layers.Dropout(0.3)(h_2)
        
        raw_output_actions = layers.Dense(units=self.n_action, activation='sigmoid')(h_2)
        
        actual_actions = layers.Lambda(lambda x: (x * (self.rotor_high - self.rotor_low)) + self.rotor_low, name='actions')(raw_output_actions)
        
        self.actorModel = models.Model(inputs = input_states, outputs = actual_actions)
        
        # action_gradients is the placeholder and it will be received from critic network 
        # and used here to update the weights of actor network
        # action_gradients is gradient of Q_value from critic network w.r.t. actions provided to critic network from actor network
        
        action_gradients = layers.Input(shape=(self.n_action,), name='action')
        
        loss = K.mean(-action_gradients*actual_actions)
        
        optimizer = optimizers.Adam()
        update_ops = optimizer.get_updates(loss, params=self.actorModel.trainable_weights)
        
        self.train_actor = K.function(inputs=[self.actorModel.input, action_gradients, K.learning_phase()], 
                                              outputs=[], updates=update_ops)
        