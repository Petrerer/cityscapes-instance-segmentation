import tensorflow as tf
import numpy as np

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
    y_pred = tf.nn.softmax(y_pred)
    
    intersection = tf.reduce_sum(y_true_onehot * y_pred, axis=[1,2])
    union = tf.reduce_sum(y_true_onehot + y_pred, axis=[1,2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

def weighted_cce(y_true, y_pred):
    class_weights = tf.constant([0.05, 0.25, 0.7])
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    weights = tf.gather(class_weights, tf.cast(y_true, tf.int32))
    return tf.reduce_mean(ce * weights)


class MeanIoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.miou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        self.miou.update_state(y_true, y_pred)

    def result(self):
        return self.miou.result()

    def reset_state(self):
        self.miou.reset_state()

def dice_coeff_metric(smooth=1e-6):
    def dice_coeff(y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        y_pred = tf.nn.softmax(y_pred)
        intersection = tf.reduce_sum(y_true_onehot * y_pred, axis=[1,2])
        union = tf.reduce_sum(y_true_onehot + y_pred, axis=[1,2])
        dice = (2. * intersection + smooth) / (union + smooth)
        return tf.reduce_mean(dice)
    return dice_coeff


class OptimizedDynamicsLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, num_classes=3, log_dir=None, 
                 max_samples=20, batch_log_freq=100):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.log_dir = log_dir
        self.max_samples = max_samples
        self.batch_log_freq = batch_log_freq
        
        self.epoch_data = []
        self.batch_losses = []
        self.batch_counter = 0
        
        if log_dir:
            self.writer = tf.summary.create_file_writer(log_dir)
        
        self._cached_val_data = None
        self._cache_validation_data()
    
    def _cache_validation_data(self):
        self._cached_val_data = None
    
    @tf.function
    def _compute_confusion_matrix(self, y_true, y_pred):
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred_labels, [-1])
        
        y_true_flat = tf.cast(y_true_flat, tf.int32)
        y_pred_flat = tf.cast(y_pred_flat, tf.int32)
        
        confusion = tf.math.confusion_matrix(
            y_true_flat, 
            y_pred_flat, 
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        
        return confusion
    
    def _calculate_class_ious_vectorized(self):
        total_confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)
        
        sample_count = 0
        batch_count = 0
        
        for x, y in self.val_dataset:
            if sample_count >= self.max_samples:
                break
            
            y_pred = self.model(x, training=False)
            
            confusion = self._compute_confusion_matrix(y, y_pred)
            
            total_confusion += confusion.numpy()
            
            del y_pred, confusion
            
            sample_count += x.shape[0]
            batch_count += 1
            
            if batch_count >= 5:
                break
        
        tp = np.diag(total_confusion)
        fp = np.sum(total_confusion, axis=0) - tp
        fn = np.sum(total_confusion, axis=1) - tp
        
        denominator = tp + fp + fn
        iou = np.where(
            denominator > 0,
            tp / (denominator + 1e-6),
            np.zeros_like(tp)
        )
        
        return iou
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        
        if hasattr(lr, 'numpy'):
            current_lr = float(lr.numpy())
        else:
            current_lr = float(lr)
        
        class_ious_np = self._calculate_class_ious_vectorized()
        
        if len(class_ious_np) > 0:
            iou_variance = float(np.var(class_ious_np))
            mean_class_iou = float(np.mean(class_ious_np))
        else:
            iou_variance = 0.0
            mean_class_iou = 0.0
        
        epoch_info = {
            'epoch': int(epoch),
            'learning_rate': current_lr,
            'class_iou_variance': iou_variance,
            'mean_class_iou': mean_class_iou,
        }
        
        for key, value in logs.items():
            if hasattr(value, 'numpy'):
                epoch_info[key] = float(value.numpy())
            else:
                epoch_info[key] = float(value)
        
        self.epoch_data.append(epoch_info)
        
        if self.log_dir and self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar('custom/learning_rate', current_lr, step=epoch)
                tf.summary.scalar('custom/class_iou_variance', iou_variance, step=epoch)
                tf.summary.scalar('custom/mean_class_iou', mean_class_iou, step=epoch)
                
                for c, iou_val in enumerate(class_ious_np):
                    tf.summary.scalar(f'custom/class_{c}_iou', float(iou_val), step=epoch)
        
        print(f"\n  LR: {current_lr:.2e} | Class IoU Var: {iou_variance:.4f} | Mean IoU: {mean_class_iou:.4f}")
    
    def on_train_batch_end(self, batch, logs=None):
        if logs and 'loss' in logs:
            loss_value = logs['loss']
            
            if hasattr(loss_value, 'numpy'):
                loss_float = float(loss_value.numpy())
            else:
                loss_float = float(loss_value)
            
            self.batch_losses.append(loss_float)
            
            if self.log_dir and self.writer is not None:
                if self.batch_counter % self.batch_log_freq == 0:
                    with self.writer.as_default():
                        tf.summary.scalar('training/batch_loss', loss_float, step=self.batch_counter)
            
            self.batch_counter += 1
    
    def export_to_json(self, filepath):
        import json
        
        export_data = {
            'epoch_metrics': self.epoch_data,
            'batch_losses': self.batch_losses,
            'config': {
                'num_classes': self.num_classes,
                'max_samples': self.max_samples,
                'batch_log_freq': self.batch_log_freq
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Metrics exported to {filepath}")


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir=None):
        super().__init__()
        self.lrs = []
        self.log_dir = log_dir
        if log_dir:
            self.writer = tf.summary.create_file_writer(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        
        # Single conversion
        lr_value = float(lr.numpy() if hasattr(lr, 'numpy') else lr)
        self.lrs.append(lr_value)
        
        # Write to TensorBoard
        if self.log_dir and self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar('learning_rate', lr_value, step=epoch)


class BatchLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir=None, log_freq=100):
        super().__init__()
        self.batch_losses = []
        self.log_freq = log_freq
        self.log_dir = log_dir
        self.batch_counter = 0
        if log_dir:
            self.writer = tf.summary.create_file_writer(log_dir)
    
    def on_train_batch_end(self, batch, logs=None):
        if logs and 'loss' in logs:
            loss_value = float(logs['loss'].numpy() if hasattr(logs['loss'], 'numpy') else logs['loss'])
            self.batch_losses.append(loss_value)
            
            # Write to TensorBoard periodically
            if self.log_dir and self.writer is not None:
                if self.batch_counter % self.log_freq == 0:
                    with self.writer.as_default():
                        tf.summary.scalar('batch_loss', loss_value, step=self.batch_counter)
            
            self.batch_counter += 1