from numpy.lib.stride_tricks import as_strided
import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from types import SimpleNamespace


# writer = SummaryWriter("logs/run4")

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


##take input the new model, and write the optimising steps
class Architect(object):
    def __init__(self, model, encoder, r_vec, args, optimizers, lr):
        # move this to train search and avoid getting in args here?
        self.lr = lr
        self.network_parameters = SimpleNamespace(
            w1=SimpleNamespace(
                network_momentum=args.momentum_w1,
                network_weight_decay=args.weight_decay_w1,
                grad_clip=args.grad_clip_w1
            ),
            w2=SimpleNamespace(
                network_momentum=args.momentum_w2,
                network_weight_decay=args.weight_decay_w2,
                grad_clip=args.grad_clip_w2
            ),
            A=SimpleNamespace(
                network_momentum=args.momentum_A,
                network_weight_decay=args.weight_decay_A,
                grad_clip=args.grad_clip_A
            ),
            V=SimpleNamespace(
                network_momentum=args.momentum_V,
                network_weight_decay=args.weight_decay_A,
                grad_clip=args.grad_clip_V
            ),
            r=SimpleNamespace(
                network_momentum=args.momentum_r,
                network_weight_decay=args.weight_decay_r,
                grad_clip=args.grad_clip_r
            ),
            train_batch_size=args.batch_size,
            val_batch_size=args.batch_size
        )
        self.optimizers = optimizers
        self.model = model  ## contains w1 w2 and A
        self.encoder = encoder  ## contains V
        self.r_vec = r_vec
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.criterion_no_reduction = nn.CrossEntropyLoss(reduction='none').cuda()
        self.args = args

    def _compute_unrolled_model_w1(self, input, target, unrolled, save_dir):  # network_w class
        if unrolled:
            # TODO: print address of assignment variables and check if they are the same
            loss = self.model.w1._loss(input, target, self.model.copy_arch_parameters())
            network_optimizer = self.optimizers.w1
            network_momentum = self.network_parameters.w1.network_momentum
            network_weight_decay = self.network_parameters.w1.network_weight_decay
            eta = self.lr.w1
            theta = _concat(self.model.w1.parameters()).data

            try:
                moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.w1.parameters()).mul_(
                    network_momentum)
            except:
                moment = torch.zeros_like(theta)
            dtheta = _concat(torch.autograd.grad(loss, self.model.w1.parameters())).data + network_weight_decay * theta
            unrolled_model = self._construct_model_from_theta_w(theta.sub(moment + dtheta, alpha=eta), save_dir)
        else:
            unrolled_model = self.model.w1
        return unrolled_model

    def _construct_model_from_theta_w(self,
                                      theta, save_dir):  ##and alpha - model.new() just initialises a new model (Network_w class)
        model_new = self.model.w1.new()
        #model_new = torch.load(os.path.join(save_dir, 'weights_model.pt')).cuda()
        model_dict = self.model.w1.state_dict()

        params, offset = {}, 0
        for k, v in self.model.w1.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    ## update this function to factor in the weights a_i
    def _compute_unrolled_model_w2(self, input, target, a_i, unrolled, save_dir):  # network_w class
        network_optimizer = self.optimizers.w2
        network_momentum = self.network_parameters.w2.network_momentum
        network_weight_decay = self.network_parameters.w2.network_weight_decay
        eta = self.lr.w2

        loss = (a_i.view(-1) * self.criterion_no_reduction(
            self.model.w2.forward(input, self.model.copy_arch_parameters()), target)).mean()
        # print('w2_train_loss', loss)
        theta = _concat(self.model.w2.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.w2.parameters()).mul_(
                network_momentum)
        except:
            moment = torch.zeros_like(theta)
        # shouldnt lose computation graph wrt w1p, V, r here in this operation
        dtheta = _concat(torch.autograd.grad(loss, self.model.w2.parameters(), create_graph=True,
                                             retain_graph=True)) + network_weight_decay * theta
        #if unrolled:
        unrolled_model = self._construct_model_from_theta_w(theta.sub(moment + dtheta.data, alpha=eta), save_dir)
        #else:
        #    unrolled_model = self.model.w2
        return unrolled_model, moment + dtheta

    def step_w1(self, input_train, target_train):
        self.optimizers.w1.zero_grad()
        logits = self.model.forward(input_train, 'w1')
        loss = self.criterion(logits, target_train)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.w1.parameters(), self.network_parameters.w1.grad_clip)
        self.optimizers.w1.step()

    def step_w2(self, input_train, target_train, input_valid, target_valid, unrolled, save_dir):
        w1_prime = self._compute_unrolled_model_w1(input_train, target_train, unrolled, save_dir)
        # TODO:check all _loss functions and replace with loss()
        l_val_w1p = self.criterion_no_reduction(w1_prime.forward(input_valid, self.model.copy_arch_parameters()),
                                                target_valid)  # --> this becomes u_i
        # TODO: verify matrix sizes and multiplication. a.shape = (n_train,)
        a_weights = torch.sigmoid(self.r_vec(
            self._compute_x(input_train, input_valid) * self._compute_z(target_train, target_valid) * l_val_w1p)).data
        self.optimizers.w2.zero_grad()
        logits = self.model.forward(input_train, 'w2')
        loss = (a_weights.view(-1) * self.criterion_no_reduction(logits, target_train)).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.w2.parameters(), self.network_parameters.w2.grad_clip)
        self.optimizers.w2.step()

    def step_AVr(self, input_train, target_train, input_valid, target_valid, unrolled, save_dir):
        self.optimizers.V.zero_grad()
        self.optimizers.r.zero_grad()
        self.optimizers.A.zero_grad()

        eta_w1 = self.lr.w1
        eta_w2 = self.lr.w2
        w1_prime = self._compute_unrolled_model_w1(input_train, target_train, unrolled, save_dir)
        l_val_w1p = self.criterion_no_reduction(w1_prime.forward(input_valid, self.model.copy_arch_parameters()),
                                                target_valid)
        a_weights = torch.sigmoid(self.r_vec(
            self._compute_x(input_train, input_valid) * self._compute_z(target_train, target_valid) * l_val_w1p))
        # print('l_val_w1p', l_val_w1p)
        # print('a_weights', a_weights)
        w2_prime, d_ailtrain_w2 = self._compute_unrolled_model_w2(input_train, target_train, a_weights, unrolled, save_dir)
        l_val_w2p = w2_prime._loss(input_valid, target_valid, self.model.arch_parameters())
        # print('l_val_w2p', l_val_w2p)

        l_val_w2p.backward()  # will update grads for w2_prime and self.model.arch_parameters
        d_lval_w2p = _concat([v.grad.data for v in w2_prime.parameters()])

        assert d_ailtrain_w2.shape == d_lval_w2p.shape
        d_ailtr_w2_dot_d_lval_w2p = torch.dot(d_ailtrain_w2, d_lval_w2p)
        # print('d_ailtr_w2_dot_d_lval_w2p', d_ailtr_w2_dot_d_lval_w2p)
        d_ailtr_w2_dot_d_lval_w2p.backward()  # has (a*xyz) -- backward for V, r, w1p
        # Encoder grads populated
        for g in self.encoder.parameters():
            g.grad.mul_(-eta_w2)
        # r grad populated
        for g in self.r_vec.parameters():
            g.grad.mul_(-eta_w2)
        d_lvalw2p_alpha = [v.grad for v in self.model.arch_parameters()]

        if unrolled:
            d_ailtr_w2_dot_d_lval_w2p__w1p = [v.grad.data for v in w1_prime.parameters()]
            finite_diff_1 = self._hessian_vector_product1(d_ailtr_w2_dot_d_lval_w2p__w1p, input_train, target_train)

            d_lval_w2p_vec = [v.grad.data for v in w2_prime.parameters()]
            finite_diff_2 = self._hessian_vector_product2(d_lval_w2p_vec, input_train, target_train,
                                                          a_weights.detach().clone())

            # caluculate implicit grads1 and 2 and populate gradients
            for g, g_fd1, g_fd2 in zip(d_lvalw2p_alpha, finite_diff_1, finite_diff_2):
                g.data.sub_(g_fd1.data.mul(-eta_w1) + g_fd2.data, alpha=eta_w2)
        else:
            d_lval_w2p_vec = [v.grad.data for v in w2_prime.parameters()]
            finite_diff_2 = self._hessian_vector_product2(d_lval_w2p_vec, input_train, target_train,
                                                          a_weights.detach().clone())

            # caluculate implicit grads1 and 2 and populate gradients
            for g, g_fd2 in zip(d_lvalw2p_alpha, finite_diff_2):
                g.data.sub_(g_fd2.data, alpha=eta_w2)

        for v, g in zip(self.model.arch_parameters(), d_lvalw2p_alpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.network_parameters.A.grad_clip)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.network_parameters.A.grad_clip)
        nn.utils.clip_grad_norm_(self.r_vec.parameters(), self.network_parameters.r.grad_clip)

        self.optimizers.V.step()
        self.optimizers.r.step()
        self.optimizers.A.step()

    ##write update for W1, W2, V, r, A in this function
    def step(self, input_train, target_train, input_valid, target_valid, unrolled, save_dir):
        self.step_AVr(input_train, target_train, input_valid, target_valid, unrolled, save_dir)
        self.step_w1(input_train, target_train)
        self.step_w2(input_train, target_train, input_valid, target_valid, unrolled, save_dir)

    def _compute_x(self, input_train, input_valid):
        train_embed = self.encoder(input_train)
        val_embed = self.encoder(input_valid)
        x = torch.matmul(train_embed, torch.transpose(val_embed, 0, 1))
        m = torch.nn.Softmax(dim=1)
        # return torch.nn.softmax()
        x = m(x)
        return x  # x_out
        # return size (ntr * nval)

    def _compute_z(self, target_train, target_valid):
        # print('target_train shape ', target_train.shape, target_valid.shape)
        z = (target_train.view(-1, 1) == target_valid).type(torch.float)
        return z
        ## return z (n_tr * n_val dim tensor)

    def _hessian_vector_product1(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        if _concat(vector).norm() == 0:
            print('norm 0 : hessian1')
            return [torch.zeros_like(x) for x in self.model.arch_parameters()]

        for p, v in zip(self.model.w1.parameters(), vector):
            p.data.add_(v, alpha=R)

        loss = self.model.loss_(input, target, 'w1')
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.w1.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)

        loss = self.model.loss_(input, target, 'w1')
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.w1.parameters(), vector):
            p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product2(self, vector, input, target, a_weights, r=1e-2):
        R = r / _concat(vector).norm()
        if _concat(vector).norm() == 0:
            print('norm 0 : hessian2')
            return [torch.zeros_like(x) for x in self.model.arch_parameters()]

        for p, v in zip(self.model.w2.parameters(), vector):
            p.data.add_(v, alpha=R)

        loss = (a_weights.view(-1) * self.criterion_no_reduction(self.model.forward(input, 'w2'), target)).mean()
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.w2.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)

        loss = (a_weights.view(-1) * self.criterion_no_reduction(self.model.forward(input, 'w2'), target)).mean()
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.w2.parameters(), vector):
            p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
