import torch 
import argparse
import pytorch_lightning as pl
import data_utils
import model
import re
from operator import add 
from functools import reduce

class Seq2SeqTrainer(pl.LightningModule): 

    def __init__(self, tokenizer: data_utils.Tokenizer) -> None: 
        super().__init__()

        encoder = model.Encoder(tokenizer.vocab_size, 128, 128, 2, 0.2)
        decoder = model.Decoder(tokenizer.vocab_size, 128, 128, 2, 0.2)

        self.tok = tokenizer
        self.net = model.Seq2Seq(encoder, decoder, self.tok.sos, self.tok.eos)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def training_step(self, batch, _batch_idx) :
        # src: (bcsz, Ts)
        # trg: (bcsz, Tt)
        
        (src, src_lengths), (trg, _trg_lengths), _ = batch

        preds = self.net(src.T, src_lengths, trg.T)
        *_, vocab_size = preds.shape

        # input: (Tt, bcsz, C) -> (bcsz, Tt, C) -> (bcsz*Tt, C)
        # target: (bcsz, Tt) -> (bcsz*Tt)

        loss = self.loss(
            input = preds[1:].permute(1, 0, 2).reshape(-1, vocab_size), 
            target = trg[:, 1:].reshape(-1)
        )

        self.log('train/loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad(): 
            (src, src_lengths), (trg, _), qas = batch
            trg = trg.cpu()

            decoded = self.net.decode(src.T, src_lengths).cpu()

            dif_history = []
            bcsz, _ = decoded.shape
            for i in range(bcsz): 
                _, a = qas[i]

                r: str = self.tok.reverse(decoded[i])
                answer_n = int(a)
                match = re.compile(r'\-?[0-9]+').search(r)
                pred_n = int(match.group()) if match is not None else 999

                dif = abs(pred_n - answer_n)
                dif_history.append(dif)


                if batch_idx == 0 and i == 0: 
                    self.logger.experiment.add_text(
                        f'decoded/text', f'{qas[i]}//{r}', 
                        global_step=self.global_step
                    )

            return dif_history
    
    def validation_epoch_end(self, outputs): 
        dif_mean = torch.Tensor(reduce(add, outputs)).mean()
        self.log('val/diff', dif_mean)

    

def main(args): 
    tok = data_utils.Tokenizer()
    collate = data_utils.ArithmeticCollate(tok.pad)
    net = Seq2SeqTrainer(tok)

    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(
        net, 
        torch.utils.data.DataLoader(data_utils.ArithmeticDataset('train.data'), batch_size=32, collate_fn=collate),
        torch.utils.data.DataLoader(data_utils.ArithmeticDataset('val.data'), batch_size=1, collate_fn=collate, shuffle=True)
    )


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
