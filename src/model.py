import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, embeddingDim=1):
        super(Encoder, self).__init__()
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
   
class Projector(tf.keras.Model):
    def __init__(self, outputDim):
        super(Projector, self).__init__()
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
       
    
class Predictor_MLP(tf.keras.models.Model):
    def __init__(self, featureLen):
        super(Predictor_MLP, self).__init__()
        num_class=2
        self.encoder = Encoder()
        self.projector = Projector(num_class)
        self.ohe = tf.keras.layers.CategoryEncoding(num_tokens=num_class, output_mode="one_hot")

        self.lossTracker = tf.keras.metrics.Mean(name="loss")
        self.BovIgvLossTracker = tf.keras.metrics.Mean(name="BOV_IGV_loss")
        self.projectorLossTracker = tf.keras.metrics.Mean(name="LEAKAGE_loss")

    @property
    def metrics(self):
        return[self.lossTracker,
               self.projectorLossTracker]
    
    def train_step(self, data):
        x,y = data
        batch_size = tf.shape(y)[0]
        leakage = tf.cast(y[:,0:1]>=0.5, tf.float32)
        leakage_prob = tf.cast(y[:,0:1], tf.float32)
        leakage_sle = tf.concat([1-y[:,0:1], y[:,0:1]],axis=-1)
        leakage_ohe = self.ohe(leakage)
        bov_igv = y[:,1:]
        bov = y[:,1:2]
        igv = y[:,2:]
        scce = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(x, training=True)
            probs = self.projector([z1, z2], training=True)
            probsLoss = tf.math.reduce_mean(scce(leakage_ohe, probs))
            valsLoss = tf.math.reduce_mean((z1-bov)**2)/2 + tf.math.reduce_mean((z2-igv)**2)/2 +\
                       tf.math.reduce_mean(abs((z1-z2)-(bov-igv)))/2
            X = tf.concat([z1,z2], axis=-1)
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
        
        trainableVars = (self.encoder.trainable_variables + 
                         self.projector.trainable_variables)
        gradients = tape.gradient(totalLoss, trainableVars,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.optimizer.apply_gradients(zip(gradients, trainableVars))

        
        self.lossTracker.update_state(totalLoss)
        self.BovIgvLossTracker.update_state(valsLoss)
        self.projectorLossTracker.update_state(probsLoss)

        return {"loss": self.lossTracker.result(),
                "BOV_IGV_loss": self.BovIgvLossTracker.result(),
                "LEAKAGE_loss": self.projectorLossTracker.result(),}
    
    def test_step(self, data):
        x,y=data
        batch_size = tf.shape(y)[0]
        leakage = tf.cast(y[:,0:1]>=0.5, tf.float32)
        leakage_prob = tf.cast(y[:,0:1], tf.float32)
        leakage_sle = tf.concat([1-y[:,0:1], y[:,0:1]],axis=-1)
        leakage_ohe = self.ohe(leakage)
        bov_igv = y[:,1:]
        bov = y[:,1:2]
        igv = y[:,2:]
        scce = tf.keras.losses.CategoricalCrossentropy()
        z1, z2 = self.encoder(x, training=False)
        probs = self.projector([z1, z2], training=False)
        probsLoss = tf.math.reduce_mean(scce(leakage_ohe, probs))
        valsLoss = tf.math.reduce_mean((z1-bov)**2)/2 + tf.math.reduce_mean((z2-igv)**2)/2 +\
                       tf.math.reduce_mean(abs((z1-z2)-(bov-igv)))/2
        X = tf.concat([z1,z2], axis=-1)
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

        self.lossTracker.update_state(totalLoss)
        self.BovIgvLossTracker.update_state(valsLoss)
        self.projectorLossTracker.update_state(probsLoss)

        return {"loss": self.lossTracker.result(),
                "BOV_IGV_loss": self.BovIgvLossTracker.result(),
                "LEAKAGE_loss": self.projectorLossTracker.result(),}
    
    def call(self, x, training=None, mask=None):
        zq1, zq2 = self.encoder(x, training=training)
        probs = self.projector([zq1, zq2], training=training)
        return [zq1, zq2], [], probs
