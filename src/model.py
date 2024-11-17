import tensorflow as tf

# Input feature 임베딩과 BOV & IGV projection을 위한 Projector
class Projector(tf.keras.Model):
    def __init__(self, embeddingDim=1):
        super(Projector, self).__init__()
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64)
        self.lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dropO1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(64)
        self.lrelu2 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dropO2 = tf.keras.layers.Dropout(0.2)
        self.denseBOV = tf.keras.layers.Dense(32)
        self.lreluBOV = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dropOBOV = tf.keras.layers.Dropout(0.2)
        self.outputsBOV = tf.keras.layers.Dense(embeddingDim)
        self.denseIGV = tf.keras.layers.Dense(32)
        self.lreluIGV = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dropOIGV = tf.keras.layers.Dropout(0.2)
        self.outputsIGV = tf.keras.layers.Dense(embeddingDim)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        z = self.flat(inputs)
        z = self.dense1(z)
        z = self.lrelu1(z)
        z = self.dropO1(z, training=training)
        z = self.dense2(z)
        z = self.lrelu2(z)
        z = self.dropO2(z, training=training)
        bov = self.denseBOV(z)
        bov = self.lreluBOV(bov)
        bov = self.dropOBOV(bov, training=training)
        igv = self.denseIGV(z)
        igv = self.lreluIGV(igv)
        igv = self.dropOIGV(igv, training=training)
        return self.outputsBOV(bov), self.outputsIGV(igv)
   
# Projected BOV & IGV로부터 air leakage probability를 예측하는 Predictor
class Predictor(tf.keras.Model):
    def __init__(self, outputDim):
        super(Predictor, self).__init__()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(8)
        self.lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dropO1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(8)
        self.lrelu2 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dropO2 = tf.keras.layers.Dropout(0.2)
        self.outputs = tf.keras.layers.Dense(outputDim, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        z = self.concat(inputs)
        z = self.dense1(z)
        z = self.lrelu1(z)
        z = self.dropO1(z, training=training)
        z = self.dense2(z)
        z = self.lrelu2(z)
        z = self.dropO2(z, training=training)
        return self.outputs(z)
       
# Projector와 Predictor로 이루어진 Air Leakage Prediction 모델
class Predictor_MLP(tf.keras.models.Model):
    def __init__(self, featureLen):
        super(Predictor_MLP, self).__init__()
        # 모델 선언
        num_class=2 # class 0: non-leakage, class 1: leakage
        self.ohe = tf.keras.layers.CategoryEncoding(num_tokens=num_class, output_mode="one_hot")
        self.projector = Projector()
        self.predictor = Predictor(num_class)
        
        # 모델의 Loss tracker 선언
        self.lossTracker = tf.keras.metrics.Mean(name="loss")
        self.BovIgvLossTracker = tf.keras.metrics.Mean(name="BOV_IGV_loss")
        self.predictorLossTracker = tf.keras.metrics.Mean(name="LEAKAGE_loss")

        # cross entropy loss 함수
        self.scce = tf.keras.losses.CategoricalCrossentropy()

    @property
    def metrics(self):
        return[self.lossTracker,
               self.predictorLossTracker]
    
    def train_step(self, data):
        # 입력 데이터 정리
        x,y = data
        batch_size = tf.shape(y)[0]
        leakage = tf.cast(y[:,0:1]>=0.5, tf.float32)
        leakage_prob = tf.cast(y[:,0:1], tf.float32)
        leakage_ohe = self.ohe(leakage)
        bov = y[:,1:2]
        igv = y[:,2:]

        # 학습
        with tf.GradientTape() as tape:

            # Feedforward
            projectedBov, projectedIgv = self.projector(x, training=True)
            probs = self.predictor([projectedBov, projectedIgv], training=True)
            
            # 실제 및 추론된 Leakage probability에 대한 Cross entropy loss
            probsLoss = tf.math.reduce_mean(self.scce(leakage_ohe, probs))
            
            # Projected된 BOV와 IGV에 대한 예측 loss (regularization)
            valsLoss = tf.math.reduce_mean((projectedBov-bov)**2)/2 + tf.math.reduce_mean((projectedIgv-igv)**2)/2 +\
                       tf.math.reduce_mean(abs((projectedBov-projectedIgv)-(bov-igv)))/2
            
            # 하나의 배치 내 동일한 class끼리 BOV와 IGV가 latent space 내 근접하게 projection이 되고
            # 다른 class끼리 BOV와 IGV가 latent space 내 멀리 위치하도록 하는 Contrastive learning loss 
            X = tf.concat([projectedBov,projectedIgv], axis=-1)
            X_norm = X/tf.norm(X,axis=-1,keepdims=True)
            X_mul = tf.matmul(X_norm, X_norm,transpose_a=False, transpose_b=True)
            nonIdentity = (tf.ones([batch_size,batch_size])-tf.eye(batch_size))
            leakage_mat = tf.math.sqrt(tf.matmul(leakage_prob,tf.transpose(leakage_prob)))*nonIdentity
            nonleakage_mat = tf.math.sqrt(tf.matmul(1-leakage_prob,tf.transpose(1-leakage_prob)))*nonIdentity
            leakage_sim = tf.math.reduce_sum(X_mul*(leakage_mat/tf.maximum(tf.math.reduce_sum(leakage_prob)-1,1)), axis=-1)
            nonleakage_sim = tf.math.reduce_sum(X_mul*(nonleakage_mat/tf.maximum(tf.math.reduce_sum(1-leakage_prob)-1,1)), axis=-1)
            dissimilar_mat = (1-leakage_mat/tf.maximum(tf.math.reduce_sum(1-leakage_prob),1)-nonleakage_mat/tf.maximum(tf.math.reduce_sum(leakage_prob)-1,1))*nonIdentity
            dissim = tf.matmul(X_mul*dissimilar_mat,1/tf.maximum(tf.math.reduce_sum(dissimilar_mat, axis=-1, keepdims=True),1))[:,0]
            simLoss = (1+dissim)/2+(1-leakage_sim)/2 +(1-nonleakage_sim)/2
            
            # leakage probability loss, latent space 내 projected BOV와 IGV에 대한 loss의 합으로 총 loss 계산
            totalLoss = probsLoss + valsLoss + simLoss
        
        # Gradient 업데이트
        trainableVars = (self.projector.trainable_variables + 
                         self.predictor.trainable_variables)
        gradients = tape.gradient(totalLoss, trainableVars,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.optimizer.apply_gradients(zip(gradients, trainableVars))

        # Loss tracking
        self.lossTracker.update_state(totalLoss)
        self.BovIgvLossTracker.update_state(valsLoss)
        self.predictorLossTracker.update_state(probsLoss)

        return {"loss": self.lossTracker.result(),
                "BOV_IGV_loss": self.BovIgvLossTracker.result(),
                "LEAKAGE_loss": self.predictorLossTracker.result(),}
    
    def test_step(self, data):
        # 입력 데이터 정리
        x,y=data
        batch_size = tf.shape(y)[0]
        leakage = tf.cast(y[:,0:1]>=0.5, tf.float32)
        leakage_prob = tf.cast(y[:,0:1], tf.float32)
        leakage_ohe = self.ohe(leakage)
        bov = y[:,1:2]
        igv = y[:,2:]

        # Feedforward
        projectedBov, projectedIgv = self.projector(x, training=False)
        probs = self.predictor([projectedBov, projectedIgv], training=False)

        # 실제 및 추론된 Leakage probability에 대한 Cross entropy loss
        probsLoss = tf.math.reduce_mean(self.scce(leakage_ohe, probs))

        # Projected된 BOV와 IGV에 대한 예측 loss (regularization)
        valsLoss = tf.math.reduce_mean((projectedBov-bov)**2)/2 + tf.math.reduce_mean((projectedIgv-igv)**2)/2 +\
                       tf.math.reduce_mean(abs((projectedBov-projectedIgv)-(bov-igv)))/2
        
        # 하나의 배치 내 동일한 class끼리 BOV와 IGV가 latent space 내 근접하게 projection이 되고
        # 다른 class끼리 BOV와 IGV가 latent space 내 멀리 위치하도록 하는 Contrastive learning loss     
        X = tf.concat([projectedBov,projectedIgv], axis=-1)
        X_norm = X/tf.norm(X,axis=-1,keepdims=True)
        X_mul = tf.matmul(X_norm, X_norm,transpose_a=False, transpose_b=True)
        nonIdentity = (tf.ones([batch_size,batch_size])-tf.eye(batch_size))
        leakage_mat = tf.math.sqrt(tf.matmul(leakage_prob,tf.transpose(leakage_prob)))*nonIdentity
        nonleakage_mat = tf.math.sqrt(tf.matmul(1-leakage_prob,tf.transpose(1-leakage_prob)))*nonIdentity
        leakage_sim = tf.math.reduce_sum(X_mul*(leakage_mat/tf.maximum(tf.math.reduce_sum(leakage_prob)-1,1)), axis=-1)
        nonleakage_sim = tf.math.reduce_sum(X_mul*(nonleakage_mat/tf.maximum(tf.math.reduce_sum(1-leakage_prob)-1,1)), axis=-1)
        dissimilar_mat = (1-leakage_mat/tf.maximum(tf.math.reduce_sum(1-leakage_prob),1)-nonleakage_mat/tf.maximum(tf.math.reduce_sum(leakage_prob)-1,1))*nonIdentity
        dissim = tf.matmul(X_mul*dissimilar_mat,1/tf.maximum(tf.math.reduce_sum(dissimilar_mat, axis=-1, keepdims=True),1))[:,0]
        simLoss = (1+dissim)/2+(1-leakage_sim)/2 +(1-nonleakage_sim)/2
        totalLoss = probsLoss+valsLoss + simLoss

        # Loss tracker
        self.lossTracker.update_state(totalLoss)
        self.BovIgvLossTracker.update_state(valsLoss)
        self.predictorLossTracker.update_state(probsLoss)

        return {"loss": self.lossTracker.result(),
                "BOV_IGV_loss": self.BovIgvLossTracker.result(),
                "LEAKAGE_loss": self.predictorLossTracker.result(),}
    
    def call(self, x, training=None, mask=None):
        # Feedforward
        projectedBov, projectedIgv = self.projector(x, training=training)
        probs = self.predictor([projectedBov, projectedIgv], training=training)
        return probs, [projectedBov, projectedIgv]
