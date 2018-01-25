import os.path


class ReportWriter:
    def __init__(self, filename, headers = []):

        self.filename = filename
        if not os.path.isfile(self.filename):
            self.file = open(self.filename, 'a')
            self.file.write(','.join(headers) + '\n')
        else:
            self.file = open(self.filename, 'a')

    def write(self, num_samples, duration, num_epochs, loss, val_loss, num_layers, cell_type, activation,
              hidden_dimension, learning_rate, gradient_clipping_value, optimizer, loss_history_filename,
              model_filename, reverse_sequence, notes):

        line = [num_samples, duration, num_epochs, loss, val_loss, num_layers, cell_type, activation, hidden_dimension,
                learning_rate, gradient_clipping_value, optimizer, loss_history_filename, model_filename,
                reverse_sequence, notes]
        line = [str(x) for x in line]

        self.file.write(','.join(line) + '\n')
