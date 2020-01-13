
import tensorflow.keras as K
import tensorflow as tf


class CustomSchedule(K.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps):
        super().__init__()
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.max_lr * tf.math.minimum(arg1, arg2) / tf.math.rsqrt(self.warmup_steps)


if __name__ == "__main__":
    temp_learning_rate_schedule = CustomSchedule(.001, 100)

    import matplotlib.pyplot as plt

    plt.plot(temp_learning_rate_schedule(tf.range(1000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    plt.show()




