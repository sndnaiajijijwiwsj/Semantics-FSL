import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args


        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass    
    
    @abc.abstractmethod
    def final_record(self):
        pass    

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            vl, va, vap = self.evaluate(self.val_loader)
            print('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, vl, va, vap))

            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model()

    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))


            print('data_timer: {:.2f} sec, '     \
                  'forward_timer: {:.2f} sec,'   \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                        self.dt.item(), self.ft.item(),
                        self.bt.item(), self.ot.item())
                  )


    def save_model(self):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.bestsave,'bestmodel-be.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
