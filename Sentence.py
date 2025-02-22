import os

class Sentence:
    def __init__(self, description = '', session_id = '', max_block_id = 0, block_id = 0,
                 trial_id = 0, num_samples = 0, sentence_text = ''):
        self.session_id = session_id.replace('.', '_')
        self.block_id = block_id
        self.desc = description
        self.idkey = f'{self.session_id}_{block_id}_{trial_id}_{description}' # unique identifier
        self.max_block_id = max_block_id
        self.num_samples = num_samples
        self.sentence_txt = sentence_text
        self.trial_id = trial_id

    def save(self, out_dir):
            # Create a file name and write a csv file.
        fname = os.path.join(out_dir + os.sep + self.idkey + '.csv')
        try:
            with open(fname, 'w') as f:
                f.write(f"{self.block_id}\n")
                f.write(f"{self.desc}\n")
                f.write(f"{self.idkey}\n")
                f.write(f"{self.max_block_id}\n")
                f.write(f"{self.num_samples}\n")
                f.write(f"{self.sentence_txt}\n")
                f.write(f"{self.session_id}\n")
                f.write(f"{self.trial_id}\n")
        except Exception as e:
            print(f"Error saving data to {fname}: {e}")
    
    def load(self, fname):
            # Read a (complete) file name and load a csv file.
        try:
            
            with open(fname, 'r') as f:
                self.block_id = f.readline().strip()
                self.desc = f.readline().strip()
                self.idkey = f.readline().strip()
                self.max_block_id = int(f.readline().strip())
                self.num_samples = int(f.readline().strip())
                self.sentence_txt = int(f.readline().strip())
                self.session_id = f.readline().strip()
                self.trial_id = int(f.readline().strip())
        except Exception as e:
            print(f"Error loading data from {fname}: {e}")
