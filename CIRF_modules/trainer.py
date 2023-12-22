import torch
import matplotlib.pyplot as plt
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import warnings; warnings.filterwarnings('ignore')
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

transform = transforms.Compose([
                                # transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])


inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def invTrans(input_tensor):
    rtn = (( input_tensor + 1 ) / 2).clamp(0, 1)
    return rtn

retransform = transforms.Compose([transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])])


tensor_to_pil_transform = transforms.ToPILImage()

class FinderTrainer():
    def __init__(self, classifier, generator, args, CTGAN_network=None, tuned_gamma=False, tabular=False, transforms=None, use_transform=False, dataset_dict=None):
        self.classifier = classifier
        self.generator = generator
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.min_shift = args.min_shift
        self.shift_scale = args.shift_scale
        self.lr = args.lr
        self.normal_method = args.normal_method
        self.num_steps = args.num_steps
        self.label_weight = args.label_weight
        self.shift_weight = args.shift_weight
        self.eps_distribution = args.eps_distribution
        self.save_path = args.save_path
        self.tensorboard_path = args.tensorboard_path
        self.k_count = args.k_count
        self.random_latent_std = args.random_latent_std
        self.tabular = tabular
        self.tuned_gamma = tuned_gamma
        self.distances_center_of_targets = None
        self.use_transform = use_transform
        self.peak_ratio = args.peak_ratio
        self.top_center = args.top_center
        self.w_space = args.w_space
        self.n_latent = args.n_latent
        self.class_range = args.class_range
        self.elastic_coef = args.elastic_coef
        self.mode_ratio = args.mode_ratio
        if self.tabular == True:
            self.CTGAN = CTGAN_network
        if transforms is None:
            self.transforms = transform
        if self.tuned_gamma == True:
            self.GMM_component = args.GMM_component
            if dataset_dict is not None:
                self.refined_latent_dict = dataset_dict
            else:
                self.refined_logits_dict, self.refined_latent_dict = self.random_refined_latent_maker(latent_iteration=args.latent_iteration, random_latent_std=args.random_latent_std, latent_batch=args.latent_batch)

            self.gmms = self.GMM_maker(refined_latent_dict=self.refined_latent_dict, gmm_component=self.GMM_component)
            self.center_of_target = self.center_of_target_maker(gmms=self.gmms)            
        
    def make_shifts(self, latent_dim, input_class, eps_distribution, tuned_gamma=False):
        target_indices = torch.randint(
            input_class * self.num_classes, (input_class + 1) * self.num_classes, [self.batch_size], device='cuda')
        if self.eps_distribution == 'uniform':
            shifts = torch.rand(target_indices.shape, device='cuda')
        elif self.eps_distribution == 'normal':
            shifts = torch.abs(torch.randn(target_indices.shape, device='cuda'))
        elif self.eps_distribution == 'none':
            shifts = torch.ones(target_indices.shape, device='cuda')

        # adopting tuned gamma, considering distances between centers of targets
        if tuned_gamma == True:
            for i, target in enumerate(target_indices):
                shifts[i] = self.distances_center_of_targets[input_class * self.num_classes + int(target)] * shifts[i] 

        elif tuned_gamma == False:
            shifts = self.shift_scale * shifts
        # shifts[(shifts < self.min_shift) & (shifts > 0)] = self.min_shift
        # shifts[(shifts > -self.min_shift) & (shifts < 0)] = -self.min_shift
        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        z_shift = torch.zeros([self.batch_size] + latent_dim, device='cuda')
        
        # for i, (index, val) in enumerate(zip(target_indices, shifts)):
        #     z_shift[i][index] += val
        
        for batch in range(self.batch_size):
            z_shift[batch] += shifts[batch]

        return target_indices, shifts, z_shift
    
    
    def make_shifts_multiclass(self, latent_dim, input_logit, eps_distribution):
        sigmoid = nn.Sigmoid()
        input_class = torch.round(sigmoid(input_logit))
        
        target_indices = torch.randint(
            0, self.num_classes, [self.batch_size], device='cuda')
        if self.eps_distribution == 'uniform':
            shifts = torch.rand(target_indices.shape, device='cuda')
        elif self.eps_distribution == 'normal':
            shifts = torch.abs(torch.randn(target_indices.shape, device='cuda'))
        elif self.eps_distribution == 'none':
            shifts = torch.ones(target_indices.shape, device='cuda')
            
        shifts = self.shift_scale * shifts
        shifts[(shifts < self.min_shift) & (shifts > 0)] = self.min_shift
        shifts[(shifts > -self.min_shift) & (shifts < 0)] = -self.min_shift
        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        z_shift = torch.zeros([self.batch_size] + latent_dim, device='cuda')
        
        # for i, (index, val) in enumerate(zip(target_indices, shifts)):
        #     z_shift[i][index] += val
        
        for batch in range(self.batch_size):
            z_shift[batch] += shifts[batch]

        return target_indices, shifts, z_shift
    
    def in_out_finder(self, target_indices):
        original_rtn = torch.zeros(target_indices.shape[0], device='cuda')
        desired_rtn = torch.zeros(target_indices.shape[0], device='cuda')
        for i, target in enumerate(target_indices):
            for decade in range(self.num_classes):
                if target >= decade * self.num_classes and target < (decade + 1) * self.num_classes:
                    original_rtn[i] = torch.tensor(decade, device='cuda')
                    desired_rtn[i] = torch.tensor(target % self.num_classes, device='cuda')
        return original_rtn, desired_rtn

    def in_out_finder_multiclass(self, pred, target_indices):
        encoded_target = torch.zeros_like(pred)
        for i, batch_iter in enumerate(encoded_target):
            encoded_target[i][target_indices[i]] = 1
        desired_class = torch.logical_xor(pred, encoded_target)
        return pred, desired_class

    def save_model(self):
        path_summery = self.save_path
        if not os.path.exists(path_summery):
            os.makedirs(path_summery)
        torch.save(self.projector.state_dict(), path_summery + '_' + self.eps_distribution + '.pt')

    def save_log(self):
        plt.subplot(4, 1, 1)
        # plt.plot(self.image_losses)
        # plt.title('image_losses')
        plt.subplot(4, 1, 2)
        plt.plot(self.counterclass_losses)
        plt.title('counterclass_losses')
        # plt.subplot(4, 1, 3)
        # plt.plot(self.shift_losses)
        # plt.title('shift_losses')
        # plt.subplot(4, 1, 4)
        # plt.plot(self.logit_losses)
        # plt.title('logit_losses')
        # plt.show()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        plt.savefig(os.path.join(self.save_path, 'losses.png'))
    
    @torch.no_grad()
    # def transform_pil(self, latent, transforms, use_transform=True):
    #     temp_batch = len(latent)
    #     rtn = []
    #     z = latent
    #     temp_generated, w = self.generator(z.to('cuda'))
    #     if use_transform == True:
    #         for i in range(temp_batch):
    #             pil_image = tensor_to_pil_transform(torchvision.utils.make_grid(temp_generated[i], normalize=True))
    #             transformed_sample = transforms(pil_image)
    #             rtn.append(transformed_sample)
    #         rtn = torch.stack(rtn)
    #         return rtn.to('cuda')
    #     elif use_transform == False:
    #         return temp_generated, w
        
    def transform_pil(self, latent, transforms, use_transform=True):
        temp_batch = len(latent)
        rtn = []
        if len(latent.shape) == 2:
            temp_generated = self.generator(latent.unsqueeze(0))
            if isinstance(temp_generated, tuple):
                temp_generated = self.generator(latent=latent.unsqueeze(0))[0]
                return temp_generated
            temp_generated = invTrans(temp_generated)
            return temp_generated
        temp_generated = self.generator(latent.to('cuda'))
        if isinstance(temp_generated, tuple):
            temp_generated = self.generator(latent=latent)[0]
            return temp_generated
        temp_generated = invTrans(temp_generated)
        if use_transform == True:
            for i in range(temp_batch):
                pil_image = tensor_to_pil_transform(torchvision.utils.make_grid(temp_generated[i], normalize=True))
                transformed_sample = transforms(pil_image)
                rtn.append(transformed_sample)
            rtn = torch.stack(rtn)
            return rtn.to('cuda')
        elif use_transform == False:
            return temp_generated
        
        
    def train(self, projector, reconstructor=None, use_reconstruction_network=True, debug_mode=False, save=True, tuned_gamma=False):
        self.projector = projector
        projector_opt = torch.optim.Adam(self.projector.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(projector_opt, T_max=self.num_steps, eta_min=self.lr*0.12)
        cross_entropy = nn.CrossEntropyLoss()

        if tuned_gamma == True:
            self.distances_center_of_targets = {}
            for from_class in range(self.num_classes):
                for to_class in range(self.num_classes):
                    idx = from_class * self.num_classes + to_class
                    if self.w_space == True:
                        distance = torch.sqrt(((self.center_of_target[from_class][0] - self.center_of_target[to_class][0])**2).sum())
                    elif self.w_space == False:
                        distance = torch.sqrt(((self.center_of_target[from_class] - self.center_of_target[to_class])**2).sum())
                    self.distances_center_of_targets[idx] = distance
                    
                    
        self.projector.train()
        self.generator.eval()
        self.classifier.eval()

        # self.logit_losses = []
        # self.shift_losses = []
        # self.image_losses = []
        # self.counterclass_losses = []
        # calculating distances between each class
        self.counterclass_losses = []

        tb_path = self.tensorboard_path
        writer = SummaryWriter(tb_path)
        pbar = tqdm(range(self.num_steps))
        for step in pbar:
            self.generator.zero_grad()
            self.projector.zero_grad()
            self.classifier.zero_grad()

            z = torch.randn(1, self.latent_dim, device='cuda') * self.random_latent_std
            imgs = self.generator(z)[0]
            
            logits_from_z = self.classifier(imgs)
            pred = int(torch.argmax(logits_from_z, 1))
            
            target_indices, gamma, basis_shift = self.make_shifts(self.latent_dim, pred, eps_distribution=self.eps_distribution, tuned_gamma=tuned_gamma)
            original_class, desired_class = self.in_out_finder(target_indices)

            input_sample = torch.zeros([self.batch_size, self.num_classes**2], device='cuda')
            for i in range(self.batch_size):
                input_sample[i][target_indices[i]] = 1
 
            shift = self.projector(input_sample)
            new_shift = torch.tensor(shift, device='cuda')
            for i in range(self.batch_size):
                new_shift[i] = shift[i] / torch.linalg.norm(shift[i]) * gamma[i]
            imgs = self.generator(z + (torch.zeros(torch.Size([self.batch_size]) + torch.Size([self.latent_dim]), device='cuda')))[0]
            imgs_shifted = self.generator(z + new_shift.reshape(torch.Size([self.batch_size]) + torch.Size([self.latent_dim])))[0]

            # image_logits, _ = self.classifier(imgs)
            shift_logits  = self.classifier(imgs_shifted)[0]

            # image_loss = cross_entropy(image_logits, original_class.long())
            counterclass_loss = cross_entropy(shift_logits, desired_class.long())

            loss = counterclass_loss
            writer.add_scalar('counterclass_loss', counterclass_loss.item(), step)
            writer.add_scalar('loss', loss.item(), step)
            loss.backward()
            projector_opt.step()

            if step % 5 == 0:
                pbar.set_description(f'counterclass_loss:{counterclass_loss.item():.4f}')
                self.counterclass_losses.append(counterclass_loss.item())
            if save==True:
                self.save_model()
        writer.close()
        
    def train_multiclass(self, projector, latent_shape, debug_mode=False, save=True):
        # self.latent_shape = latent_shape
        self.projector = projector
        projector_opt = torch.optim.Adam(self.projector.parameters(), lr=self.lr)
        cross_entropy = nn.CrossEntropyLoss()

        self.projector.train()
        self.generator.eval()
        self.classifier.eval()

        # self.logit_losses = []
        # self.shift_losses = []
        # self.image_losses = []
        self.counterclass_losses = []

        tb_path = self.tensorboard_path
        writer = SummaryWriter(tb_path)
        pbar = tqdm(range(self.num_steps))
        criterion = nn.BCELoss()
        for step in pbar:
            self.generator.zero_grad()
            self.projector.zero_grad()
            self.classifier.zero_grad()

            z = torch.randn([self.batch_size, self.latent_dim], device='cuda')
            imgs = self.generator(z)[0]
            logits_from_z = self.classifier(imgs)
            sigmoid = nn.Sigmoid()
            pred = torch.round(sigmoid(logits_from_z))
            
            target_indices, shifts, basis_shift = self.make_shifts_multiclass(self.latent_dim, logits_from_z, eps_distribution=self.eps_distribution)
            original_class, desired_class = self.in_out_finder_multiclass(pred.detach(), target_indices)

            input_sample = torch.zeros([self.batch_size, self.num_classes], device='cuda')
            for i in range(self.batch_size):
                input_sample[i][target_indices[i]] = 1
            
            from_target_class = []
            for i in range(self.batch_size): # concating original, desired class
                from_target_class.append(torch.cat([original_class[i], desired_class[i]]))
            from_target_class = torch.stack(from_target_class).to('cuda')
                
            shift = self.projector(from_target_class.float())
            new_shift = torch.tensor(shift, device='cuda')
            for i in range(self.batch_size):
                new_shift[i] = shift[i] / torch.linalg.norm(shift[i]) * shifts[i]
            # imgs = self.generator(z + (torch.zeros(torch.Size([self.batch_size]) + self.latent_dim[1:], device='cuda')))
            # imgs_shifted = self.generator(z + new_shift.reshape(torch.Size([self.batch_size]) + self.latent_dim[1:]))
            imgs = self.generator(z + torch.zeros(torch.Size([self.batch_size]) + torch.Size([self.latent_dim]), device='cuda'))[0]
            imgs_shifted = self.generator(z + new_shift)[0]

            image_logits = self.classifier(imgs)
            shift_logits = self.classifier(imgs_shifted)
            shift_pred = sigmoid(shift_logits)

            # image_loss = cross_entropy(image_logits, original_class.long())
            counterclass_loss = criterion(shift_pred, desired_class.float())

            loss = counterclass_loss
            writer.add_scalar('counterclass_loss', counterclass_loss.item(), step)
            writer.add_scalar('loss', loss.item(), step)
            loss.backward()
            projector_opt.step()
            if step % 5 == 0:
                pbar.set_description(f'counterclass_loss:{counterclass_loss.item():.4f}')
                self.counterclass_losses.append(counterclass_loss.item())

            if step % 50 == 0:
                pbar.set_description(f'counterclass_loss:{counterclass_loss:.4f}')

                # self.logit_losses.append(logit_loss)
                # self.shift_losses.append(shift_loss)
                # self.image_losses.append(image_loss)
                # self.counterclass_losses.append(counterclass_loss)

            # if step % 500 == 0 and debug_mode == True :
            #     plt.plot(self.image_losses)
            #     plt.title('image_losses')
            #     plt.show()
            #     plt.plot(self.counterclass_losses)
            #     plt.title('counterclass_losses')
            #     plt.show()
            #     plt.plot(self.shift_losses)
            #     plt.title('shift_losses')
            #     plt.show()
            #     plt.plot(self.logit_losses)
            #     plt.title('logit_losses')
            #     plt.show()
            #     print(f'step:{step}')

                self.save_log()
            if save==True:
                self.save_model()
        writer.close()
        
    def train_celebA(self, projector, latent_shape, debug_mode=False, save=True, tuned_gamma=False, w_space=False):
        # self.latent_shape = latent_shape
        self.projector = projector
        projector_opt = torch.optim.Adam(self.projector.parameters(), lr=self.lr)
        cross_entropy = nn.CrossEntropyLoss()

        if tuned_gamma == True:
            self.distances_center_of_targets = {}
            for from_class in range(self.num_classes):
                for to_class in range(self.num_classes):
                    idx = from_class * self.num_classes + to_class
                    if self.w_space == True:
                        distance = torch.sqrt(((self.center_of_target[from_class][0] - self.center_of_target[to_class][0])**2).mean())
                    elif self.w_space == False:
                        distance = torch.sqrt(((self.center_of_target[from_class] - self.center_of_target[to_class])**2).mean())
                    self.distances_center_of_targets[idx] = distance
                    
        self.projector.train()
        self.generator.eval()
        self.classifier.eval()

        # self.logit_losses = []
        # self.shift_losses = []
        # self.image_losses = []
        self.counterclass_losses = []

        tb_path = self.tensorboard_path
        writer = SummaryWriter(tb_path)
        pbar = tqdm(range(self.num_steps))
        
        for step in pbar:
            self.generator.zero_grad()
            self.projector.zero_grad()
            self.classifier.zero_grad()

            z = torch.randn([1, self.latent_dim], device='cuda') * self.random_latent_std
            if self.use_transform == True:
                imgs, w = self.transform_pil(z, transforms=transform, use_transform=self.use_transform)
            elif self.use_transform == False:
                imgs, w = self.generator(z)
            logits_from_z = self.classifier(imgs)
            pred = int(torch.argmax(logits_from_z, 1))
            
            target_indices, gamma, basis_shift = self.make_shifts(self.latent_dim, pred, eps_distribution=self.eps_distribution, tuned_gamma=tuned_gamma)
            original_class, desired_class = self.in_out_finder(target_indices)

            input_sample = torch.zeros([self.batch_size, self.k_count], device='cuda')
            for i in range(self.batch_size):
                input_sample[i][target_indices[i]] = 1

            shift = self.projector(input_sample)
            # new_shift = shift
            new_shift = torch.tensor(shift, device='cuda')
            for i in range(self.batch_size):
                new_shift[i] = shift[i] / torch.linalg.norm(shift[i]) * gamma[i]
            if w_space == True:
                new_shift = new_shift.unsqueeze(1)
            # imgs = self.generator(z + (torch.zeros(torch.Size([self.batch_size]) + self.latent_dim[1:], device='cuda')))
            # imgs_shifted = self.generator(z + new_shift.reshape(torch.Size([self.batch_size]) + self.latent_dim[1:]))
            # imgs = self.transform_pil(z + torch.zeros(torch.Size([self.batch_size]) + torch.Size([self.latent_dim]), device='cuda'), transforms=transform, use_transform=self.use_transform)
            # imgs = self.generator(z + torch.zeros(torch.Size([self.batch_size]) + torch.Size([self.latent_dim]), device='cuda'))[0]
            # imgs_shifted = self.generator(z + new_shift)[0]
            if self.use_transform == True : 
                imgs_shifted = self.transform_pil(z + new_shift, transforms=transform, use_transform=self.use_transform)
                # matching transform (imagenet arguements -> 0.5 ...)
                inv_transed_shifted = inv_trans(imgs_shifted)
                imgs_shifted = retransform(inv_transed_shifted)
            elif self.use_transform == False:
                if w_space == True:
                    imgs_shifted = self.generator(latent=w + new_shift)[0]
                else:
                    imgs_shifted = self.generator(z + new_shift)[0]
            
            # image_logits = self.classifier(imgs)
            shift_logits = self.classifier(imgs_shifted)

            # shift_pred = torch.argmax(shift_logits, 1).float()
            # image_loss = cross_entropy(image_logits, original_class.long())
            counterclass_loss = cross_entropy(shift_logits, desired_class.long())

            loss = counterclass_loss
            writer.add_scalar('counterclass_loss', counterclass_loss.item(), step)
            writer.add_scalar('loss', loss.item(), step)
            loss.backward()
            projector_opt.step()
            if step % 2 == 0:
                pbar.set_description(f'counterclass_loss:{counterclass_loss.item():.4f}')
                self.counterclass_losses.append(counterclass_loss.item())

            # if step % 50 == 0:
            #     pbar.set_description(f'counterclass_loss:{counterclass_loss:.4f}')

                # self.logit_losses.append(logit_loss)
                # self.shift_losses.append(shift_loss)
                # self.image_losses.append(image_loss)
                # self.counterclass_losses.append(counterclass_loss)

            # if step % 500 == 0 and debug_mode == True :
            #     plt.plot(self.image_losses)
            #     plt.title('image_losses')
            #     plt.show()
            #     plt.plot(self.counterclass_losses)
            #     plt.title('counterclass_losses')
            #     plt.show()
            #     plt.plot(self.shift_losses)
            #     plt.title('shift_losses')
            #     plt.show()
            #     plt.plot(self.logit_losses)
            #     plt.title('logit_losses')
            #     plt.show()
            #     print(f'step:{step}')

                self.save_log()
            if save==True:
                self.save_model()
        writer.close()
        

    def train_ImageNet_w_one(self, projector, input_latent, latent_shape, debug_mode=False, save=True, tuned_gamma=False, w_space=True, dataloader=None, class_range=0):
        # self.latent_shape = latent_shape
        # self.projector = projector
        # init_direction = torch.randn(1, input_latent.shape[2]).to('cuda').requires_grad_(True)
        # init_direction = nn.Parameter(torch.zeros(1, input_latent.shape[2], device='cuda'), requires_grad=True)
        if self.w_space==True:
            init_direction = nn.Parameter(torch.randn(1, input_latent.shape[2], device='cuda'), requires_grad=True)
        else:
            init_direction = nn.Parameter(torch.randn(1, input_latent.shape[1], device='cuda'), requires_grad=True)
        projector_opt = torch.optim.Adam([init_direction], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(projector_opt, T_max=self.num_steps, eta_min=self.lr*0.12)
        # projector_opt = torch.optim.Adam(list(self.projector.parameters()) + [init_noise], lr=self.lr)
        cross_entropy = nn.CrossEntropyLoss()
        
        first_class = int(self.class_range.split('-')[0])
        second_class = int(self.class_range.split('-')[1])
        
        label_dict = {first_class:0,
                      second_class:1}
        rev_label_dict = {0:first_class,
                          1:second_class}

        if tuned_gamma == True:
            self.distances_center_of_targets = {}
            for from_class in range(self.num_classes):
                for to_class in range(self.num_classes):
                    idx = from_class * self.num_classes + to_class
                    if self.w_space == True:
                        distance = torch.sqrt(((self.center_of_target[from_class][0] - self.center_of_target[to_class][0])**2).mean())
                    elif self.w_space == False:
                        distance = torch.sqrt(((self.center_of_target[from_class] - self.center_of_target[to_class])**2).mean())
                    self.distances_center_of_targets[idx] = distance
                    
        # self.projector.train()
        
        self.generator.eval()
        self.classifier.eval()

        # self.logit_losses = []
        # self.shift_losses = []
        # self.image_losses = []
        self.counterclass_losses = []

        tb_path = self.tensorboard_path
        writer = SummaryWriter(tb_path)
        pbar = tqdm(range(self.num_steps))
        for step in pbar:
            self.generator.zero_grad()
            # self.projector.zero_grad()
            self.classifier.zero_grad()
            z = input_latent
            # imgs = self.generator(z)
            # imgs = invTrans(imgs)
            # logits_from_z = self.classifier(imgs)
            # pred = int(torch.argmax(logits_from_z, 1))
            # if pred not in label_dict.keys():
            #     # print(f'unexpected class:{pred}, {step}')
            #     continue
            # else:
            #     pred = label_dict[pred]
            
            norm_factor = torch.linalg.norm(init_direction)
            normed_init_direction = init_direction / norm_factor
            
            shift = z + self.distances_center_of_targets[1] * normed_init_direction
            # shift = z + init_direction
            imgs_shifted = self.generator(shift)
            # imgs_shifted = ((imgs_shifted + 1) * (255/2)).clamp(0, 255) / 255 # erased the clamp to preserve the gradients
            imgs_shifted = ((imgs_shifted + 1) * (255/2)) / 255
            imgs_shifted = transform(imgs_shifted)
            shift_logits = self.classifier(imgs_shifted)
            counterclass_loss = cross_entropy(shift_logits, torch.tensor([second_class]).to('cuda').long()) 
            # counterclass_loss = cross_entropy(shift_logits, torch.tensor([second_class]).to('cuda').long())
            l1_loss = torch.norm(init_direction, dim=1, p=1) 
            l2_loss = torch.norm(init_direction, dim=1, p=2) 
            # loss = counterclass_loss
            loss = counterclass_loss + self.elastic_coef * l1_loss * 0.1 + self.elastic_coef * l2_loss
            writer.add_scalar('counterclass_loss', counterclass_loss.item(), step)
            writer.add_scalar('loss', loss.item(), step)
            loss.backward()
            projector_opt.step()
            scheduler.step()
            if step % 2 == 0:
                pbar.set_description(f'total_loss:{loss.item():.4f}, cf:{counterclass_loss.item():.4f}, direction_norm:{torch.norm(init_direction, p=1):.4f}, {torch.norm(init_direction, p=2):.4f}')
                self.counterclass_losses.append(counterclass_loss.item())
            if save==True:
                self.save_log()
        writer.close()
        return init_direction, self.counterclass_losses
    
    def train_ImageNet_w(self, projector, latent_shape, debug_mode=False, save=True, tuned_gamma=False, w_space=True, dataloader=None, class_range=0):
        # self.latent_shape = latent_shape
        self.projector = projector
        projector_opt = torch.optim.Adam(self.projector.parameters(), lr=self.lr)
        cross_entropy = nn.CrossEntropyLoss()
        
        first_class = int(self.class_range.split('-')[0])
        second_class = int(self.class_range.split('-')[1])
        
        label_dict = {first_class:0,
                      second_class:1}
        rev_label_dict = {0:first_class,
                          1:second_class}

        if tuned_gamma == True:
            self.distances_center_of_targets = {}
            for from_class in range(self.num_classes):
                for to_class in range(self.num_classes):
                    idx = from_class * self.num_classes + to_class
                    if self.w_space == True:
                        distance = torch.sqrt(((self.center_of_target[from_class][0] - self.center_of_target[to_class][0])**2).mean())
                    elif self.w_space == False:
                        distance = torch.sqrt(((self.center_of_target[from_class] - self.center_of_target[to_class])**2).mean())
                    self.distances_center_of_targets[idx] = distance
                    
        self.projector.train()
        self.generator.eval()
        self.classifier.eval()

        # self.logit_losses = []
        # self.shift_losses = []
        # self.image_losses = []
        self.counterclass_losses = []

        tb_path = self.tensorboard_path
        writer = SummaryWriter(tb_path)
        
        for epoch in range(self.num_steps):
            pbar = tqdm(dataloader)
            for step, data in enumerate(pbar):
                self.generator.zero_grad()
                self.projector.zero_grad()
                self.classifier.zero_grad()
                w, label = data['input'].to('cuda'), data['label'].to('cuda')
                z = torch.randn([1, self.latent_dim], device='cuda') * self.random_latent_std
                imgs = self.generator(w)
                imgs = invTrans(imgs)
                logits_from_z = self.classifier(imgs)
                pred = int(torch.argmax(logits_from_z, 1))
                if pred not in label_dict.keys():
                    print(f'unexpected class:{pred}, {step}')
                    continue
                else:
                    pred = label_dict[pred]
                target_indices, gamma, basis_shift = self.make_shifts(self.latent_dim, pred, eps_distribution=self.eps_distribution, tuned_gamma=tuned_gamma)
                original_class, desired_class = self.in_out_finder(target_indices)
                desired_class = [rev_label_dict[int(desired_class[batch_iter])] for batch_iter in range(self.batch_size)]
                desired_class = torch.tensor(desired_class, device='cuda')
                input_sample = torch.zeros([self.batch_size, self.k_count], device='cuda')
                for i in range(self.batch_size):
                    input_sample[i][target_indices[i]] = 1

                shift = self.projector(input_sample)
                # new_shift = shift
                new_shift = torch.tensor(shift, device='cuda')
                for i in range(self.batch_size):
                    new_shift[i] = shift[i] / torch.linalg.norm(shift[i]) * gamma[i]
                    # new_shift[i] = shift[i]  * gamma[i]
                if w_space == True:
                    new_shift = new_shift.unsqueeze(1)
                if self.use_transform == True : 
                    imgs_shifted = self.transform_pil(w + new_shift, transforms=transform, use_transform=self.use_transform)
                    # matching transform (imagenet arguements -> 0.5 ...)
                    inv_transed_shifted = inv_trans(imgs_shifted)
                    imgs_shifted = retransform(inv_transed_shifted)
                elif self.use_transform == False:
                    if w_space == True:
                        imgs_shifted = self.generator(w + new_shift)
                    else:
                        imgs_shifted = self.generator(z + new_shift)
                imgs_shifted = invTrans(imgs_shifted)
                shift_logits = self.classifier(imgs_shifted)
                counterclass_loss = cross_entropy(shift_logits, desired_class.long())

                loss = counterclass_loss
                writer.add_scalar('counterclass_loss', counterclass_loss.item(), step)
                writer.add_scalar('loss', loss.item(), step)
                loss.backward()
                projector_opt.step()
                if step % 2 == 0:
                    pbar.set_description(f'counterclass_loss:{counterclass_loss.item():.4f}')
                    self.counterclass_losses.append(counterclass_loss.item())
                    self.save_log()
                if save==True:
                    self.save_model()
            writer.close()
            
    def train_celebA_w(self, projector, latent_shape, debug_mode=False, save=True, tuned_gamma=False, w_space=True, dataloader=None):
        # self.latent_shape = latent_shape
        self.projector = projector
        projector_opt = torch.optim.Adam(self.projector.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(projector_opt, T_max=self.num_steps, eta_min=self.lr*0.12)
        cross_entropy = nn.CrossEntropyLoss()

        if tuned_gamma == True:
            self.distances_center_of_targets = {}
            for from_class in range(self.num_classes):
                for to_class in range(self.num_classes):
                    idx = from_class * self.num_classes + to_class
                    if self.w_space == True:
                        distance = torch.sqrt(((self.center_of_target[from_class][0] - self.center_of_target[to_class][0])**2).sum())
                    elif self.w_space == False:
                        distance = torch.sqrt(((self.center_of_target[from_class] - self.center_of_target[to_class])**2).sum())
                    self.distances_center_of_targets[idx] = distance
                    
        self.projector.train()
        self.generator.eval()
        self.classifier.eval()

        # self.logit_losses = []
        # self.shift_losses = []
        # self.image_losses = []
        self.counterclass_losses = []

        tb_path = self.tensorboard_path
        writer = SummaryWriter(tb_path)
        
        for epoch in range(100):
            pbar = tqdm(dataloader)
            for step, data in enumerate(pbar):
                self.generator.zero_grad()
                self.projector.zero_grad()
                self.classifier.zero_grad()
                w, label = data['input'].to('cuda'), data['label'].to('cuda')
                z = torch.randn([1, self.latent_dim], device='cuda') * self.random_latent_std
                # if self.use_transform == True:
                #     imgs, w = self.transform_pil(z, transforms=transform, use_transform=self.use_transform)
                # elif self.use_transform == False:
                #     imgs, w = self.generator(z)
                imgs = self.generator(latent=w)[0]
                logits_from_z = self.classifier(imgs)
                pred = int(torch.argmax(logits_from_z, 1))
                
                target_indices, gamma, basis_shift = self.make_shifts(self.latent_dim, pred, eps_distribution=self.eps_distribution, tuned_gamma=tuned_gamma)
                original_class, desired_class = self.in_out_finder(target_indices)

                input_sample = torch.zeros([self.batch_size, self.k_count], device='cuda')
                for i in range(self.batch_size):
                    input_sample[i][target_indices[i]] = 1

                shift = self.projector(input_sample)
                # new_shift = shift
                new_shift = torch.tensor(shift, device='cuda')
                for i in range(self.batch_size):
                    new_shift[i] = shift[i] / torch.linalg.norm(shift[i]) * gamma[i]
                    # new_shift[i] = shift[i] * gamma[i]
                if w_space == True:
                    new_shift = new_shift.unsqueeze(1)
                # imgs = self.generator(z + (torch.zeros(torch.Size([self.batch_size]) + self.latent_dim[1:], device='cuda')))
                # imgs_shifted = self.generator(z + new_shift.reshape(torch.Size([self.batch_size]) + self.latent_dim[1:]))
                # imgs = self.transform_pil(z + torch.zeros(torch.Size([self.batch_size]) + torch.Size([self.latent_dim]), device='cuda'), transforms=transform, use_transform=self.use_transform)
                # imgs = self.generator(z + torch.zeros(torch.Size([self.batch_size]) + torch.Size([self.latent_dim]), device='cuda'))[0]
                # imgs_shifted = self.generator(z + new_shift)[0]
                if self.use_transform == True : 
                    imgs_shifted = self.transform_pil(w + new_shift, transforms=transform, use_transform=self.use_transform)
                    # matching transform (imagenet arguements -> 0.5 ...)
                    inv_transed_shifted = inv_trans(imgs_shifted)
                    imgs_shifted = retransform(inv_transed_shifted)
                elif self.use_transform == False:
                    if w_space == True:
                        imgs_shifted = self.generator(latent=w + new_shift)[0]
                    else:
                        imgs_shifted = self.generator(z + new_shift)[0]
                
                # image_logits = self.classifier(imgs)
                shift_logits = self.classifier(imgs_shifted)

                # shift_pred = torch.argmax(shift_logits, 1).float()
                # image_loss = cross_entropy(image_logits, original_class.long())
                counterclass_loss = cross_entropy(shift_logits, desired_class.long())

                loss = counterclass_loss
                writer.add_scalar('counterclass_loss', counterclass_loss.item(), step)
                writer.add_scalar('loss', loss.item(), step)
                loss.backward()
                projector_opt.step()
                if step % 2 == 0:
                    pbar.set_description(f'counterclass_loss:{counterclass_loss.item():.4f}')
                    self.counterclass_losses.append(counterclass_loss.item())

                # if step % 50 == 0:
                #     pbar.set_description(f'counterclass_loss:{counterclass_loss:.4f}')

                    # self.logit_losses.append(logit_loss)
                    # self.shift_losses.append(shift_loss)
                    # self.image_losses.append(image_loss)
                    # self.counterclass_losses.append(counterclass_loss)

                # if step % 500 == 0 and debug_mode == True :
                #     plt.plot(self.image_losses)
                #     plt.title('image_losses')
                #     plt.show()
                #     plt.plot(self.counterclass_losses)
                #     plt.title('counterclass_losses')
                #     plt.show()
                #     plt.plot(self.shift_losses)
                #     plt.title('shift_losses')
                #     plt.show()
                #     plt.plot(self.logit_losses)
                #     plt.title('logit_losses')
                #     plt.show()
                #     print(f'step:{step}')

                    self.save_log()
                if save==True:
                    self.save_model()
            writer.close()


    def train_celebA_w_one(self, input_latent, original_class, target_class, latent_shape, debug_mode=False, save=False, tuned_gamma=False, w_space=True, dataloader=None):
        # self.latent_shape = latent_shape
        # for i in range(30):
        if self.w_space==True:
            CT_initvec = self.center_of_target[target_class] - self.center_of_target[original_class]
            init_direction = nn.Parameter(CT_initvec.unsqueeze(0), requires_grad=True)
        else:
            CT_initvec = self.center_of_target[target_class] - self.center_of_target[original_class]
            init_direction = nn.Parameter(CT_initvec.unsqueeze(0), requires_grad=True)
            # init_direction = nn.Parameter(torch.zeros(1, input_latent.shape[1], device='cuda'), requires_grad=True)
        projector_opt = torch.optim.Adam([init_direction], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(projector_opt, T_max=self.num_steps, eta_min=self.lr*0.12)
        cross_entropy = nn.CrossEntropyLoss()
        
        first_class = original_class
        second_class = target_class

        if tuned_gamma == True:
            self.distances_center_of_targets = {}
            for from_class in range(self.num_classes):
                for to_class in range(self.num_classes):
                    idx = from_class * self.num_classes + to_class
                    if self.w_space == True:
                        distance = torch.sqrt(((self.center_of_target[from_class][0] - self.center_of_target[to_class][0])**2).sum())
                    elif self.w_space == False:
                        distance = torch.sqrt(((self.center_of_target[from_class] - self.center_of_target[to_class])**2).sum())
                    self.distances_center_of_targets[idx] = distance
                    
        self.generator.eval()
        self.classifier.eval()

        # self.logit_losses = []
        # self.shift_losses = []
        # self.image_losses = []
        self.counterclass_losses = []

        tb_path = self.tensorboard_path
        writer = SummaryWriter(tb_path)
        for resume in range(3):
            pbar = tqdm(range(self.num_steps))
            for step in pbar:
                second_class = target_class

                self.generator.zero_grad()
                # self.projector.zero_grad()
                self.classifier.zero_grad()
                z = input_latent.clone().detach()
                # imgs = self.generator(z)
                # imgs = invTrans(imgs)
                # logits_from_z = self.classifier(imgs)
                # pred = int(torch.argmax(logits_from_z, 1))
                # if pred not in label_dict.keys():
                #     # print(f'unexpected class:{pred}, {step}')
                #     continue
                # else:
                #     pred = label_dict[pred]
                
                norm_factor = torch.linalg.norm(init_direction)
                normed_init_direction = init_direction / norm_factor + 1e-6
                
                shift = z + self.distances_center_of_targets[int(original_class + self.num_classes * target_class)] * normed_init_direction
                # shift = z + init_direction
                if self.w_space==True:
                    imgs_shifted = self.generator(latent=shift)[0]
                    shift_logits = self.classifier(imgs_shifted)
                else:
                    imgs_shifted = self.generator(shift)
                    shift_logits = self.classifier(imgs_shifted)
                    
                # imgs_shifted = inv_trans(imgs_shifted)
                counterclass_loss = cross_entropy(shift_logits, torch.tensor([second_class]).to('cuda').long()) 
                # counterclass_loss = cross_entropy(shift_logits, torch.tensor([second_class]).to('cuda').long())
                if self.w_space==True:
                    l1_loss = torch.norm(init_direction, dim=2, p=1).mean()
                    l2_loss = torch.norm(init_direction, dim=2, p=2).mean()
                else:
                    l1_loss = torch.norm(init_direction, dim=1, p=1) 
                    l2_loss = torch.norm(init_direction, dim=1, p=2) 
                # loss = counterclass_loss
                loss = counterclass_loss + self.elastic_coef * l1_loss * 0.1 + self.elastic_coef * l2_loss
                writer.add_scalar('counterclass_loss', counterclass_loss.item(), step)
                writer.add_scalar('loss', loss.item(), step)
                loss.backward()
                projector_opt.step()
                scheduler.step()
                if step % 2 == 0:
                    pbar.set_description(f'total_loss:{loss.item():.4f}, cf:{counterclass_loss.item():.4f}, direction_norm:{torch.norm(init_direction, p=1):.4f}, {torch.norm(init_direction, p=2):.4f}')
                    self.counterclass_losses.append(counterclass_loss.item())
                if save==True:
                    self.save_log()
            writer.close()
            if counterclass_loss.item() <= 1:
                return init_direction, self.counterclass_losses
        return init_direction, self.counterclass_losses


    def generate_activate_transform(self, input, CTGAN, only_activate=False):
        gen_data = CTGAN._generator(input)
        activated_data = CTGAN._apply_activate(gen_data)
        if only_activate==True:
            return activated_data
        elif only_activate==False:
            inv_transformed_data = CTGAN._transformer.inverse_transform(activated_data.detach().cpu().numpy())
            return inv_transformed_data, activated_data


    def train_tabular(self, input_latent, original_class, target_class, CTGAN_network, tuned_gamma=True):
        if self.w_space==True:
            CT_initvec = self.center_of_target[target_class] - self.center_of_target[original_class]
            init_direction = nn.Parameter(CT_initvec.unsqueeze(0), requires_grad=True)
        else:
            CT_initvec = self.center_of_target[target_class] - self.center_of_target[original_class]
            init_direction = nn.Parameter(CT_initvec.unsqueeze(0), requires_grad=True)
        projector_opt = torch.optim.Adam([init_direction], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(projector_opt, T_max=self.num_steps, eta_min=self.lr*0.12)
        cross_entropy = nn.CrossEntropyLoss()
        
        first_class = original_class
        second_class = target_class
        
        if tuned_gamma == True:
            self.distances_center_of_targets = {}
            for from_class in range(self.num_classes):
                for to_class in range(self.num_classes):
                    idx = from_class * self.num_classes + to_class
                    if self.w_space == True:
                        distance = torch.sqrt(((self.center_of_target[from_class][0] - self.center_of_target[to_class][0])**2).sum())
                    elif self.w_space == False:
                        distance = torch.sqrt(((self.center_of_target[from_class] - self.center_of_target[to_class])**2).sum())
                    self.distances_center_of_targets[idx] = distance

        self.generator.eval()
        self.classifier.eval()
        self.counterclass_losses = []

        tb_path = self.tensorboard_path
        writer = SummaryWriter(tb_path)
        for resume in range(3):
            pbar = tqdm(range(self.num_steps))
            for step in pbar:
                second_class = target_class
                self.generator.zero_grad()
                self.classifier.zero_grad()
                
                z = input_latent.clone().detach()
                norm_factor = torch.linalg.norm(init_direction)
                normed_init_direction = init_direction / norm_factor + 1e-6

                shift = z + self.distances_center_of_targets[int(original_class + self.num_classes * target_class)] * normed_init_direction
            
                if self.w_space==True:
                    imgs_shifted = self.generator(latent=shift)[0]
                    shift_logits = self.classifier(imgs_shifted)
                else:
                    if self.tabular==True:
                        gen_data, activated_data = self.generate_activate_transform(input=shift, CTGAN=CTGAN_network)
                        shift_logits = self.classifier(activated_data)
                    else:
                        imgs_shifted = self.generator(shift)
                        shift_logits = self.classifier(imgs_shifted)
                               
                counterclass_loss = cross_entropy(shift_logits, torch.tensor([second_class]).to('cuda').long())
                # tb_path = os.path.join(self.tensorboard_path, 'tb_results/loss')
                if self.w_space==True:
                    l1_loss = torch.norm(init_direction, dim=2, p=1).mean()
                    l2_loss = torch.norm(init_direction, dim=2, p=2).mean()
                else:
                    l1_loss = torch.norm(init_direction, dim=1, p=1) 
                    l2_loss = torch.norm(init_direction, dim=1, p=2) 

                loss = counterclass_loss + self.elastic_coef * l1_loss * 0.1 + self.elastic_coef * l2_loss
                writer.add_scalar('counterclass_loss', counterclass_loss.item(), step)
                writer.add_scalar('loss', loss.item(), step)
                loss.backward()
                projector_opt.step()
                scheduler.step()
                if step % 2 == 0:
                    pbar.set_description(f'total_loss:{loss.item():.4f}, cf:{counterclass_loss.item():.4f}, direction_norm:{torch.norm(init_direction, p=1):.4f}, {torch.norm(init_direction, p=2):.4f}')
                    self.counterclass_losses.append(counterclass_loss.item())
                    self.save_log()

            writer.close()
            if counterclass_loss.item() <= 1:
                return init_direction, self.counterclass_losses
        return init_direction, self.counterclass_losses


    @torch.no_grad()
    def random_refined_latent_maker(self, latent_iteration, random_latent_std=None, latent_batch=None):
        if random_latent_std is None:
            random_latent_std = self.random_latent_std
        self.logits_dict = {}
        self.latent_dict = {}

        if self.tabular==False:
            if self.w_space == True:
                z_sum = torch.zeros([latent_batch * latent_iteration, self.n_latent, self.latent_dim])
            elif self.w_space == False:
                z_sum = torch.zeros([latent_batch * latent_iteration, self.latent_dim])    
        elif self.tabular==True:
            z_sum = torch.zeros([latent_batch * latent_iteration, self.latent_dim])

        logits_from_z = []
        
        for iter in tqdm(range(latent_iteration), desc='latent generating'):
            if self.tabular==False:
                if self.w_space == True:
                    z = random_latent_std * torch.randn([latent_batch, 1, self.latent_dim]).to('cuda')
                    z = self.generator(z, return_only_latent=True)
                    z = z.repeat(1, self.n_latent, 1)
                else:
                    z = random_latent_std * torch.randn([latent_batch, self.latent_dim]).to('cuda')
            elif self.tabular==True:
                z = random_latent_std * torch.randn([latent_batch, self.latent_dim])
            if self.tabular==False:
                if self.w_space == True:
                    transformed_sample = self.transform_pil(z, transforms=self.transforms, use_transform=self.use_transform)
                elif self.w_space == False:
                    transformed_sample = self.generator(z)
                logits_from_z.append(self.classifier(transformed_sample).to('cpu'))
            elif self.tabular==True:
                activated_data = self.generate_activate_transform(z.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
                logits_from_z.append(self.classifier(activated_data).to('cpu'))
            if self.w_space == True:
                z_sum[iter * latent_batch : (iter + 1) * latent_batch] = z
            elif self.w_space == False:
                z_sum[iter * latent_batch : (iter + 1) * latent_batch] = z


        logits_from_z = torch.cat(logits_from_z)
        pred = torch.argmax(logits_from_z, 1)
        for i in range(self.num_classes):
            self.logits_dict[i] = logits_from_z[torch.where(pred == i)]
            self.latent_dict[i] = z_sum[torch.where(pred == i)]
        
        self.refined_logits_dict = {}
        self.refined_latent_dict = {}
        for i in range(self.num_classes):
            top_idx = self.logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(self.logits_dict[i]) * self.peak_ratio)]
            self.refined_logits_dict[i] = self.logits_dict[i][top_idx]
            self.refined_latent_dict[i] = self.latent_dict[i][top_idx]
        return self.refined_logits_dict, self.refined_latent_dict
    
        
        # if latent_batch <= 50:
        #     z_sum = torch.zeros([latent_batch])
        #     for cumul in range(30):
        #         for iter in range(latent_iteration):
        #             if self.tabular==False:
        #                 z = random_latent_std * torch.randn([latent_batch, self.latent_dim])
        #             elif self.tabular==True:
        #                 z = random_latent_std * torch.randn([latent_batch, self.latent_dim])
        #             z_sum[iter * latent_batch : (iter + 1) * latent_batch] = z
        #             if self.tabular==False:
        #                 logits_from_z.append(self.classifier(self.generator(z.to('cuda'))[0]).to('cpu'))
        #             elif self.tabular==True:
        #                 activated_data = self.generate_activate_transform(z.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
        #                 logits_from_z.append(self.classifier(activated_data).to('cpu'))
        #     latent_batch = latent_batch * 30

        # for iter in tqdm(range(latent_iteration), desc='latent generating'):
        #     if self.tabular==False:
        #         z = random_latent_std * torch.randn([latent_batch, self.latent_dim]).to('cuda')
        #     elif self.tabular==True:
        #         z = random_latent_std * torch.randn([latent_batch, self.latent_dim]).to('cuda')
        #     if self.tabular==False:
        #         transformed_sample, w = self.transform_pil(z, transforms=self.transforms, use_transform=self.use_transform)
        #         logits_from_z.append(self.classifier(transformed_sample).to('cpu'))
        #     elif self.tabular==True:
        #         activated_data = self.generate_activate_transform(z.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
        #         logits_from_z.append(self.classifier(activated_data).to('cpu'))
        #     if self.w_space == True:
        #         z_sum[iter * latent_batch : (iter + 1) * latent_batch] = w
        #     elif self.w_space == False:
        #         z_sum[iter * latent_batch : (iter + 1) * latent_batch] = z

        # logits_from_z = torch.cat(logits_from_z)
        # pred = torch.argmax(logits_from_z, 1)
        # for i in range(self.num_classes):
        #     self.logits_dict[i] = logits_from_z[torch.where(pred == i)]
        #     self.latent_dict[i] = z_sum[torch.where(pred == i)]
        
        # refined_logits_dict = {}
        # refined_latent_dict = {}
        # for i in range(self.num_classes):
        #     top_idx = self.logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(self.logits_dict[i]) * self.peak_ratio)]
        #     refined_logits_dict[i] = self.logits_dict[i][top_idx]
        #     refined_latent_dict[i] = self.latent_dict[i][top_idx]
        # return refined_logits_dict, refined_latent_dict
    
    def center_of_target_maker(self, gmms):

        if self.normal_method == 'median':
            target_mode = []
            for i in range(self.num_classes):
                target_mode.append(self.refined_latent_dict[i].median(0)[0])
            target_mode = torch.stack(target_mode).to('cuda')

        elif self.normal_method == 'mean':
            target_mode = []
            for i in range(self.num_classes):
                target_mode.append(self.refined_latent_dict[i].mean(0))
            target_mode = torch.stack(target_mode).to('cuda')
            
        elif self.normal_method == 'mode':
            if self.w_space == True:
                target_mode = torch.zeros([self.num_classes, self.n_latent, self.latent_dim], device='cuda')
            elif self.w_space == False:
                target_mode = torch.zeros([self.num_classes, self.latent_dim], device='cuda')
            
            for from_class in tqdm(range(self.num_classes), desc='center of target maker'):
                for element in range(self.latent_dim):
                    sample_element = gmms[from_class][element].sample(10000)
                    probabiliy_gmm = gmms[from_class][element].score_samples(sample_element[0])
                    sorted_indices = np.argsort(probabiliy_gmm, axis=0)
                    max_idx = sorted_indices[0 if self.mode_ratio==0 else round(10000 * self.mode_ratio) - 1]
                    if self.w_space == True:
                        temp_mode = torch.tensor(sample_element[0][max_idx]).repeat(self.n_latent)
                        target_mode[from_class][:, element] = temp_mode
                    elif self.w_space == False:
                        target_mode[from_class][element] = torch.tensor(sample_element[0][max_idx])
        
        return target_mode
        
    def GMM_maker(self, refined_latent_dict, gmm_component):
        gmms = {}
        for gmm_class in tqdm(range(self.num_classes), desc='gmm training'):
            gmms_element = {}
            # if refined_latent_dict[gmm_class].shape[0] < self.GMM_component:
            #     for element in range(self.latent_dim):
            #         gmm = GaussianMixture(n_components=gmm_component, random_state=0)
            #         gmm = gmm.fit(np.zeros(self.latent_dim).reshape(-1, 1))
            #         gmms_element[element] = gmm
            #     gmms[gmm_class] = gmms_element
            #     continue
            
            for element in range(self.latent_dim):
                gmm = GaussianMixture(n_components=gmm_component, random_state=0)
                if self.w_space == True:
                    gmm = gmm.fit(refined_latent_dict[gmm_class][:, 0, element].reshape(-1, 1).detach().cpu().numpy())
                elif self.w_space == False:
                    gmm = gmm.fit(refined_latent_dict[gmm_class][:, element].reshape(-1, 1).detach().cpu().numpy())
                gmms_element[element] = gmm
            gmms[gmm_class] = gmms_element
        return gmms

    