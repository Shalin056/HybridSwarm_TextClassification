# import numpy as np
# import tensorflow as tf
# from models import build_lstm_model, build_cnn_model, build_bert_model

# class PSO:
#     def __init__(self, n_particles, bounds, model_type, input_dim, output_dim, X_train, y_train, X_val, y_val):
#         """
#         Initialize PSO for hyperparameter optimization.
#         Args:
#             n_particles (int): Number of particles in the swarm.
#             bounds (list): [(min, max)] for each hyperparameter.
#             model_type (str): 'lstm', 'cnn', or 'bert'.
#             input_dim (int): Input dimension for the model.
#             output_dim (int): Number of classes.
#             X_train, y_train: Training data and labels.
#             X_val, y_val: Validation data and labels.
#         """
#         self.n_particles = n_particles
#         self.bounds = bounds
#         self.n_dims = len(bounds)
#         self.model_type = model_type
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_val = X_val
#         self.y_val = y_val
        
#         # Initialize particles' positions and velocities
#         self.positions = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_particles, self.n_dims))
#         self.velocities = np.zeros((n_particles, self.n_dims))
#         self.pbest_positions = self.positions.copy()
#         self.pbest_scores = np.full(n_particles, -np.inf)
#         self.gbest_position = self.positions[0].copy()
#         self.gbest_score = -np.inf

#     def evaluate(self, position):
#         """Evaluate a particle's position (hyperparameters) by training a model."""
#         if self.model_type == 'lstm':
#             model = build_lstm_model(self.input_dim, self.output_dim, lstm_units=int(position[0]), dropout_rate=position[1])
#             history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)  # Changed to 1 epoch
#             val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
#         elif self.model_type == 'cnn':
#             model = build_cnn_model(self.input_dim, self.output_dim, filters=int(position[0]), kernel_size=int(position[1]), dropout_rate=position[2])
#             history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
#             val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
#         elif self.model_type == 'bert':
#             model = build_bert_model(trainable=False, learning_rate=position[0])
#             history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
#             val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
#         return val_acc

#     def optimize(self, max_iter, w=0.7, c1=2, c2=2):
#         """Run PSO optimization."""
#         for _ in range(max_iter):
#             for i in range(self.n_particles):
#                 # Evaluate current position
#                 score = self.evaluate(self.positions[i])
                
#                 # Update personal best
#                 if score > self.pbest_scores[i]:
#                     self.pbest_scores[i] = score
#                     self.pbest_positions[i] = self.positions[i].copy()
                
#                 # Update global best
#                 if score > self.gbest_score:
#                     self.gbest_score = score
#                     self.gbest_position = self.positions[i].copy()
            
#             # Update velocities and positions
#             r1, r2 = np.random.rand(2)
#             self.velocities = (w * self.velocities + 
#                                c1 * r1 * (self.pbest_positions - self.positions) + 
#                                c2 * r2 * (self.gbest_position - self.positions))
#             self.positions = self.positions + self.velocities
            
#             # Clamp positions to bounds
#             for d in range(self.n_dims):
#                 self.positions[:, d] = np.clip(self.positions[:, d], self.bounds[d][0], self.bounds[d][1])
        
#         return self.gbest_position, self.gbest_score

# class GWO:
#     def __init__(self, n_wolves, bounds, model_type, input_dim, output_dim, X_train, y_train, X_val, y_val):
#         """Initialize GWO for hyperparameter optimization."""
#         self.n_wolves = n_wolves
#         self.bounds = bounds
#         self.n_dims = len(bounds)
#         self.model_type = model_type
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_val = X_val
#         self.y_val = y_val
        
#         # Initialize wolves' positions
#         self.positions = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_wolves, self.n_dims))
#         self.alpha_pos = self.positions[0].copy()
#         self.beta_pos = self.positions[0].copy()
#         self.delta_pos = self.positions[0].copy()
#         self.alpha_score = -np.inf
#         self.beta_score = -np.inf
#         self.delta_score = -np.inf

#     def evaluate(self, position):
#         """Evaluate a wolf's position (same as PSO)."""
#         if self.model_type == 'lstm':
#             model = build_lstm_model(self.input_dim, self.output_dim, lstm_units=int(position[0]), dropout_rate=position[1])
#             history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)  # Changed to 1 epoch
#             val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
#         elif self.model_type == 'cnn':
#             model = build_cnn_model(self.input_dim, self.output_dim, filters=int(position[0]), kernel_size=int(position[1]), dropout_rate=position[2])
#             history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
#             val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
#         elif self.model_type == 'bert':
#             model = build_bert_model(trainable=False, learning_rate=position[0])
#             history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
#             val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
#         return val_acc

#     def optimize(self, max_iter):
#         """Run GWO optimization."""
#         for t in range(max_iter):
#             a = 2 - 2 * t / max_iter  # Linearly decrease a from 2 to 0
            
#             for i in range(self.n_wolves):
#                 score = self.evaluate(self.positions[i])
                
#                 # Update alpha, beta, delta
#                 if score > self.alpha_score:
#                     self.delta_score = self.beta_score
#                     self.delta_pos = self.beta_pos.copy()
#                     self.beta_score = self.alpha_score
#                     self.beta_pos = self.alpha_pos.copy()
#                     self.alpha_score = score
#                     self.alpha_pos = self.positions[i].copy()
#                 elif score > self.beta_score:
#                     self.delta_score = self.beta_score
#                     self.delta_pos = self.beta_pos.copy()
#                     self.beta_score = score
#                     self.beta_pos = self.positions[i].copy()
#                 elif score > self.delta_score:
#                     self.delta_score = score
#                     self.delta_pos = self.positions[i].copy()
            
#             # Update positions
#             for i in range(self.n_wolves):
#                 for d in range(self.n_dims):
#                     r1, r2 = np.random.rand(2)
#                     A1 = 2 * a * r1 - a
#                     C1 = 2 * r2
#                     D_alpha = abs(C1 * self.alpha_pos[d] - self.positions[i, d])
#                     X1 = self.alpha_pos[d] - A1 * D_alpha
                    
#                     r1, r2 = np.random.rand(2)
#                     A2 = 2 * a * r1 - a
#                     C2 = 2 * r2
#                     D_beta = abs(C2 * self.beta_pos[d] - self.positions[i, d])
#                     X2 = self.beta_pos[d] - A2 * D_beta
                    
#                     r1, r2 = np.random.rand(2)
#                     A3 = 2 * a * r1 - a
#                     C3 = 2 * r2
#                     D_delta = abs(C3 * self.delta_pos[d] - self.positions[i, d])
#                     X3 = self.delta_pos[d] - A3 * D_delta
                    
#                     self.positions[i, d] = (X1 + X2 + X3) / 3
                
#                 # Clamp to bounds
#                 self.positions[i] = np.clip(self.positions[i], [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
#         return self.alpha_pos, self.alpha_score
    
# class HybridPSOGWO:
#     def __init__(self, n_agents, bounds, model_type, input_dim, output_dim, X_train, y_train, X_val, y_val):
#         self.n_agents = n_agents
#         self.bounds = bounds
#         self.n_dims = len(bounds)
#         self.model_type = model_type
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_val = X_val
#         self.y_val = y_val
        
#         # Initialize with PSO
#         self.pso = PSO(n_agents, bounds, model_type, input_dim, output_dim, X_train, y_train, X_val, y_val)
#         self.positions = self.pso.positions.copy()
#         self.best_position = None
#         self.best_score = -np.inf

#     def evaluate(self, position):
#         """Same evaluation function as PSO/GWO."""
#         if self.model_type == 'lstm':
#             model = build_lstm_model(self.input_dim, self.output_dim, lstm_units=int(position[0]), dropout_rate=position[1])
#             history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)  # Changed to 1 epoch
#             val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
#         return val_acc
    
#     def optimize(self, max_iter):
#         """Hybrid optimization: PSO for half iterations, GWO for the rest."""
#         half_iter = max_iter // 2
        
#         # PSO phase
#         self.pso.optimize(half_iter)
#         self.positions = self.pso.positions.copy()
#         self.best_position = self.pso.gbest_position.copy()
#         self.best_score = self.pso.gbest_score
        
#         # GWO phase
#         gwo = GWO(self.n_agents, self.bounds, self.model_type, self.input_dim, self.output_dim, 
#                   self.X_train, self.y_train, self.X_val, self.y_val)
#         gwo.positions = self.positions.copy()  # Start with PSO's final positions
#         best_pos, best_score = gwo.optimize(max_iter - half_iter)
        
#         if best_score > self.best_score:
#             self.best_position = best_pos
#             self.best_score = best_score
        
#         return self.best_position, self.best_score



#v2

import numpy as np
import tensorflow as tf
from models import build_lstm_model, build_cnn_model, build_bert_model

class PSO:
    def __init__(self, n_particles, bounds, model_type, input_dim, output_dim, X_train, y_train, X_val, y_val):
        self.n_particles = n_particles
        self.bounds = bounds  # e.g., [(32, 128), (0.2, 0.5), (16, 64)] for lstm_units, dropout_rate, batch_size
        self.n_dims = len(bounds)
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Initialize particles
        self.positions = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_particles, self.n_dims))
        self.velocities = np.zeros((n_particles, self.n_dims))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(n_particles, -np.inf)
        self.gbest_position = self.positions[0].copy()
        self.gbest_score = -np.inf

    def evaluate(self, position):
        if self.model_type == 'lstm':
            model = build_lstm_model(self.input_dim, self.output_dim, 
                                     lstm_units=int(position[0]), dropout_rate=position[1])
            history = model.fit(self.X_train, self.y_train, epochs=2, batch_size=int(position[2]), verbose=0)
            val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
        elif self.model_type == 'cnn':
            model = build_cnn_model(self.input_dim, self.output_dim, 
                                    filters=int(position[0]), kernel_size=int(position[1]), dropout_rate=position[2])
            history = model.fit(self.X_train, self.y_train, epochs=2, batch_size=32, verbose=0)
            val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
        elif self.model_type == 'bert':
            model = build_bert_model(trainable=False)  # Fixed for simplicity; could optimize batch_size
            history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=int(position[0]), verbose=0)
            val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
        return val_acc

    def optimize(self, max_iter, w=0.7, c1=2, c2=2):
        for _ in range(max_iter):
            for i in range(self.n_particles):
                score = self.evaluate(self.positions[i])
                if score > self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                if score > self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()
            
            r1, r2 = np.random.rand(2)
            self.velocities = (w * self.velocities + 
                               c1 * r1 * (self.pbest_positions - self.positions) + 
                               c2 * r2 * (self.gbest_position - self.positions))
            self.positions = self.positions + self.velocities
            for d in range(self.n_dims):
                self.positions[:, d] = np.clip(self.positions[:, d], self.bounds[d][0], self.bounds[d][1])
        
        return self.gbest_position, self.gbest_score

class GWO:
    def __init__(self, n_wolves, bounds, model_type, input_dim, output_dim, X_train, y_train, X_val, y_val):
        self.n_wolves = n_wolves
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.positions = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_wolves, self.n_dims))
        self.alpha_pos = self.positions[0].copy()
        self.beta_pos = self.positions[0].copy()
        self.delta_pos = self.positions[0].copy()
        self.alpha_score = -np.inf
        self.beta_score = -np.inf
        self.delta_score = -np.inf

    def evaluate(self, position):
        if self.model_type == 'lstm':
            model = build_lstm_model(self.input_dim, self.output_dim, 
                                     lstm_units=int(position[0]), dropout_rate=position[1])
            history = model.fit(self.X_train, self.y_train, epochs=2, batch_size=int(position[2]), verbose=0)
            val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
        elif self.model_type == 'cnn':
            model = build_cnn_model(self.input_dim, self.output_dim, 
                                    filters=int(position[0]), kernel_size=int(position[1]), dropout_rate=position[2])
            history = model.fit(self.X_train, self.y_train, epochs=2, batch_size=32, verbose=0)
            val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
        elif self.model_type == 'bert':
            model = build_bert_model(trainable=False)
            history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=int(position[0]), verbose=0)
            val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=0)
        return val_acc

    def optimize(self, max_iter):
        for t in range(max_iter):
            a = 2 - 2 * t / max_iter
            for i in range(self.n_wolves):
                score = self.evaluate(self.positions[i])
                if score > self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = score
                    self.alpha_pos = self.positions[i].copy()
                elif score > self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = score
                    self.beta_pos = self.positions[i].copy()
                elif score > self.delta_score:
                    self.delta_score = score
                    self.delta_pos = self.positions[i].copy()
            
            for i in range(self.n_wolves):
                for d in range(self.n_dims):
                    r1, r2 = np.random.rand(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[d] - self.positions[i, d])
                    X1 = self.alpha_pos[d] - A1 * D_alpha
                    
                    r1, r2 = np.random.rand(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[d] - self.positions[i, d])
                    X2 = self.beta_pos[d] - A2 * D_beta
                    
                    r1, r2 = np.random.rand(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[d] - self.positions[i, d])
                    X3 = self.delta_pos[d] - A3 * D_delta
                    
                    self.positions[i, d] = (X1 + X2 + X3) / 3
                self.positions[i] = np.clip(self.positions[i], [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        return self.alpha_pos, self.alpha_score

class HybridPSOGWO:
    def __init__(self, n_agents, bounds, model_type, input_dim, output_dim, X_train, y_train, X_val, y_val):
        self.n_agents = n_agents
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.pso = PSO(n_agents, bounds, model_type, input_dim, output_dim, X_train, y_train, X_val, y_val)
        self.positions = self.pso.positions.copy()
        self.best_position = None
        self.best_score = -np.inf

    def optimize(self, max_iter):
        half_iter = max_iter // 2
        self.pso.optimize(half_iter)
        self.positions = self.pso.positions.copy()
        self.best_position = self.pso.gbest_position.copy()
        self.best_score = self.pso.gbest_score
        
        gwo = GWO(self.n_agents, self.bounds, self.model_type, self.input_dim, self.output_dim, 
                  self.X_train, self.y_train, self.X_val, self.y_val)
        gwo.positions = self.positions.copy()
        best_pos, best_score = gwo.optimize(max_iter - half_iter)
        
        if best_score > self.best_score:
            self.best_position = best_pos
            self.best_score = best_score
        
        return self.best_position, self.best_score