import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
import os
from sklearn.svm import SVC
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from torchvision import transforms


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])])

imagenetTransform = transforms.Compose([
                                # transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), 
                                                    (0.229, 0.224, 0.225))
                                ])

# def invTrans(input_tensor):
#     rtn = (( input_tensor + 1 ) / 2).clamp(0, 1)
#     return rtn

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

tensor_to_pil_transform = transforms.ToPILImage()


class FinderMaker_ImageNet_Plausible():
    def __init__(self, projector, classifier, generator, args, CTGAN=None, original_image=None, auged_data=None, tabular=False, transforms=None, gmms=None, center_of_target=None, SVM_coefs=None):
        # self.finder_coef = 1
        self.latent_dim = args.latent_dim
        self.k_count = args.k_count
        self.num_classes = args.num_classes
        # self.projector = projector
        self.classifier = classifier
        self.generator = generator
        self.GMM_component = args.GMM_component
        self.peak_ratio = args.peak_ratio
        self.tabular = tabular
        self.use_transform = args.use_transform
        self.top_center = args.top_center
        self.w_space = args.w_space
        self.n_latent =  args.n_latent
        self.transforms = transforms

        if CTGAN:
            self.CTGAN = CTGAN
        if args.latent_type == 'training':
            self.auged_data = auged_data
            self.original_image = original_image
            self.logits, self.labels = self.logit_label_finder()
            self.refined_logits_dict, self.refined_latent_dict = self.refined_latent_maker()
        elif args.latent_type == 'random':
            self.latent_batch = args.latent_batch
            if args.w_traindata_path != None:
                self.refined_latent_dict = torch.load(args.w_traindata_path)
            else:
                self.refined_logits_dict, self.refined_latent_dict = self.random_refined_latent_maker(latent_iteration=args.latent_iteration, random_latent_std=args.random_latent_std, latent_batch=self.latent_batch)
        if self.top_center == True:
            self.center_of_target = self.center_of_target_maker(gmms=False)
        elif self.top_center == False:
            if gmms == None and center_of_target == None:
                self.gmms = self.GMM_maker(refined_latent_dict=self.refined_latent_dict, gmm_component=self.GMM_component)
                self.center_of_target = self.center_of_target_maker(gmms=self.gmms)
            else:
                self.gmms = gmms
                self.center_of_target = center_of_target
        if SVM_coefs==None:
            self.SVM_coefs = self.SVM_coef_maker()
        else:
            self.SVM_coefs = SVM_coefs
        # self.finder_coefs = self.finder_coef_maker()
        self.normal_method = args.normal_method
        self.path = args.path
        self.sub_weight = args.sub_weight      
        self.input_latent_path = args.input_latent_path
        self.random_latent_std = args.random_latent_std
        self.modifying_trial = args.modifying_trial
        self.use_calibration = args.use_calibration
        self.use_perturbing = args.use_perturbing
        self.class_label_dict, self.rev_class_label_dict = self.target_range_maker(args.class_range)
        
    def target_range_maker(self, class_range):
        first_class = int(class_range.split('-')[0])
        second_class = int(class_range.split('-')[1])

        class_label_dict = {
                            first_class:0,
                            second_class:1
                            }
        rev_class_label_dict = {
                            0:first_class,
                            1:second_class
                            }
        return class_label_dict, rev_class_label_dict


    @torch.no_grad()
    def transform_pil(self, latent, transforms, use_transform=True):
        temp_batch = len(latent)
        rtn = []
        if self.w_space==False:
            return self.generator(latent)
        if len(latent.shape) == 2:
            temp_generated = self.generator(latent.unsqueeze(0))
            if isinstance(temp_generated, tuple):
                temp_generated = self.generator(latent=latent.unsqueeze(0))[0]
                return temp_generated
            temp_generated = invTrans(temp_generated)
            return temp_generated
        if self.w_space == True:
            temp_generated = self.generator(latent.to('cuda'))
            if isinstance(temp_generated, tuple):
                temp_generated = self.generator(latent=latent)[0]
                return temp_generated
            temp_generated = ((temp_generated + 1) * (255/2)).clamp(0, 255) / 255
            temp_generated = imagenetTransform(temp_generated)
            return temp_generated
        else:
            temp_generated = self.generator(latent.to('cuda'))

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
        
    def refined_latent_maker(self):
        logits_dict = {}
        latent_dict = {}
        with torch.no_grad():
            logits, _ = self.classifier(self.original_image / 255 * 2 - 1)
        for i in range(self.num_classes):
            logits_dict[i] = logits[torch.where(self.labels == i)]
            latent_dict[i] = self.auged_data[torch.where(self.labels == i)]

        refined_logits_dict = {}
        refined_latent_dict = {}
        for i in range(self.num_classes):
            top_idx = logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(logits_dict[i]) * self.peak_ratio)]
            refined_logits_dict[i] = logits_dict[i][top_idx]
            refined_latent_dict[i] = latent_dict[i][top_idx]
        return refined_logits_dict, refined_latent_dict

    def generate_activate_transform(self, input, CTGAN, only_activate=False):
        gen_data = CTGAN._generator(input)
        activated_data = CTGAN._apply_activate(gen_data)
        if only_activate==True:
            return activated_data
        elif only_activate==False:
            inv_transformed_data = CTGAN._transformer.inverse_transform(activated_data.detach().cpu().numpy())
            return inv_transformed_data, activated_data
        

    @torch.no_grad()
    def random_refined_latent_maker(self, latent_iteration, random_latent_std=None, latent_batch=None):
        if random_latent_std == None:
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
                    z = z.repeat(1, self.n_latent, 1)
                else:
                    z = random_latent_std * torch.randn([latent_batch, self.latent_dim]).to('cuda')
            elif self.tabular==True:
                z = random_latent_std * torch.randn([latent_batch, self.latent_dim]).to('cuda')
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
                z_sum[iter * latent_batch : (iter + 1) * latent_batch] = w
            elif self.w_space == False:
                z_sum[iter * latent_batch : (iter + 1) * latent_batch] = z


        logits_from_z = torch.cat(logits_from_z)
        pred = torch.argmax(logits_from_z, 1)
        for i in range(self.num_classes):
            self.logits_dict[i] = logits_from_z[torch.where(pred == i)]
            self.latent_dict[i] = z_sum[torch.where(pred == i)]
        
        refined_logits_dict = {}
        refined_latent_dict = {}
        for i in range(self.num_classes):
            top_idx = self.logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(self.logits_dict[i]) * self.peak_ratio)]
            refined_logits_dict[i] = self.logits_dict[i][top_idx]
            refined_latent_dict[i] = self.latent_dict[i][top_idx]
        return refined_logits_dict, refined_latent_dict
    
    def SVM_coef_maker(self):
        if self.tabular==False:
            if self.w_space == True:
                SVM_coefs = torch.zeros([self.k_count, 14, self.latent_dim])
            elif self.w_space == False:
                SVM_coefs = torch.zeros([self.k_count, 1, self.latent_dim])
        if self.tabular==True:
            SVM_coefs = torch.zeros([self.k_count, 1, self.latent_dim])
        
        SVM_coefs = []
        for from_class in tqdm(range(self.num_classes), desc='SVM coef maker'):
            for to_class in range(self.num_classes):
                svc_model = SVC(C=1.0, kernel='linear')
                if self.w_space == True:
                    sample_origin_target_data = torch.cat([self.refined_latent_dict[from_class][:, 0], self.refined_latent_dict[to_class][:, 0]])
                elif self.w_space == False:
                    sample_origin_target_data = torch.cat([self.refined_latent_dict[from_class], self.refined_latent_dict[to_class]])
                # if len(self.refined_latent_dict[from_class]) == 0 or len(self.refined_latent_dict[to_class]) == 0:
                #     temp_direction = torch.zeros(1, self.latent_dim)
                #     SVM_coefs.append(temp_direction.to('cuda'))
                #     continue
                sample_origin_target_label = torch.cat([torch.zeros(len(self.refined_latent_dict[from_class])), torch.ones(len(self.refined_latent_dict[to_class]))])
                sample_origin_target_data = sample_origin_target_data.detach().cpu().numpy()
                sample_origin_target_label = sample_origin_target_label.detach().cpu().numpy()

                svc_model.fit(sample_origin_target_data.squeeze(), sample_origin_target_label)
                temp_coef = svc_model.coef_ / np.linalg.norm(svc_model.coef_)
                temp_coef = temp_coef.astype(float)
                if self.tabular==False:
                    if self.w_space == True:
                        temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)
                        temp_coef = temp_coef.repeat(self.n_latent, 1)
                    elif self.w_space == False:
                        temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)
                elif self.tabular==True:
                    temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)

                SVM_coefs.append(temp_coef)
                # SVM_coefs[torch.stack([from_class, to_class])] = temp_coef
        SVM_coefs = torch.stack(SVM_coefs)
        return SVM_coefs.to('cuda')

    def finder_coef_maker(self):
        if self.tabular==False:
            finder_coefs = torch.zeros([self.k_count, 1, self.latent_dim], device='cuda')
        elif self.tabular==True:
            finder_coefs = torch.zeros([self.k_count, 1, self.latent_dim], device='cuda')
            
        finder_coefs = []
        for idx in range(self.k_count):
            sample = torch.zeros(self.k_count, device='cuda')
            sample[idx] = 1
            sample = sample.to('cuda')
            with torch.no_grad():
                if self.tabular==False:
                    temp_finder_coef = self.projector(sample).reshape(1, self.latent_dim)
                elif self.tabular==True:
                    temp_finder_coef = self.projector(sample).reshape(1, self.latent_dim)
                temp_finder_coef = temp_finder_coef / torch.linalg.norm(temp_finder_coef)

                finder_coefs.append(temp_finder_coef)
        finder_coefs = torch.stack(finder_coefs)
        if self.w_space == True:
            return finder_coefs.repeat(1, self.n_latent, 1).to('cuda')
        elif self.w_space == False:
            return finder_coefs.to('cuda')

    def logit_label_finder(self):
        input_image = self.original_image.to('cpu')
        cpu_classifier = self.classifier.to('cpu')
        logits = cpu_classifier(input_image / 255 * 2 - 1)[0]
        softmax_func = torch.nn.Softmax(dim=1)
        softmax_result = softmax_func(logits)
        labels = softmax_result.argmax(1)
        logits = logits.to('cuda')
        labels = labels.to('cuda')
        return logits, labels

    def direction_maker(self, z, original_class, target_class, coef_name, target_range=0, direction=None):
        if coef_name == 'svm':
            coef = self.SVM_coefs[original_class * self.num_classes + target_class]
        elif coef_name == 'finder':
            coef = self.finder_coefs[original_class * self.num_classes + target_class]
        elif coef_name == 'one':
            coef = direction
        scalar=0
        
        # defining linspace(limit_dist) for interpolating
        class_dists = torch.zeros(self.num_classes ** 2)
        for from_class in range(self.num_classes):
            for to_class in range(self.num_classes):
                if self.w_space == True:
                    class_dist = self.center_of_target[from_class][0] - self.center_of_target[to_class][0]
                else :
                    class_dist = self.center_of_target[from_class] - self.center_of_target[to_class]
                class_dist_norm = torch.sqrt((class_dist ** 2).sum())
                class_dists[from_class * self.num_classes + to_class] = class_dist_norm
        max_dist = class_dists.max()
        limit_dist = torch.round(max_dist * 1.5)

        z_direction, z_semifactual, scalar = self.latent_finder(latent=z, 
                                                coef=coef, 
                                                target_class=target_class, 
                                                recent_scalar=scalar, 
                                                scalar_start=0, 
                                                scalar_end=limit_dist, 
                                                num=101, 
                                                debug_mode=False,
                                                target_range=self.class_label_dict)

        with torch.no_grad():
            if self.tabular==False:
                if self.w_space==True:
                    transformed_sample = self.transform_pil(z_direction, transforms=self.transforms, use_transform=self.use_transform)
                    temp_logits = self.classifier(transformed_sample)
                elif self.w_space==False:
                    transformed_sample = self.generator(z_direction)
                    temp_logits = self.classifier(transformed_sample)
            elif self.tabular==True:
                temp_activated = self.generate_activate_transform(z_direction.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
                temp_logits = self.classifier(temp_activated)
            temp_pred = int(torch.argmax(temp_logits))
        if temp_pred == self.rev_class_label_dict[1]: # for one algorithm
        # if temp_pred == self.rev_class_label_dict[1]: # for k algorithm
            return z_direction, z_semifactual
            
        print('not found in direction')
        return z_direction, z_semifactual

    @torch.no_grad()
    def latent_finder(self, latent, coef, target_class, recent_scalar, target_range=0, scalar_start=0, scalar_end=10, num=1000, debug_mode=False):
        z = latent
        for scalar in tqdm(np.linspace(scalar_start, scalar_end, num), desc='direction_maker'):
            if scalar == 0:
                continue
            scalar = torch.tensor(scalar, dtype=float).to('cuda')
            scalar_new = scalar + recent_scalar
            # print(f'scalar:{scalar_new}')
            if self.tabular == False:
                if self.w_space == True:
                    temp_latent = (z.to('cuda') + (scalar_new) * coef).float()
                    transformed_sample = self.transform_pil(temp_latent, transforms=self.transforms, use_transform=self.use_transform)
                elif self.w_space == False:
                    temp_latent = (z.to('cuda') + (scalar_new) * coef.reshape(1, self.latent_dim)).float()
                    transformed_sample = self.generator(temp_latent)
                logits = self.classifier(transformed_sample)
            elif self.tabular == True:
                temp_latent = (z.to('cuda') + (scalar_new) * coef.reshape(1, self.latent_dim)).float()
                temp_activated = self.generate_activate_transform(temp_latent, self.CTGAN, only_activate=True)
                logits = self.classifier(temp_activated)
            if int(torch.argmax(logits, 1)) not in target_range.keys():
                continue
            pred = int(torch.argmax(logits, 1))

            if pred == self.rev_class_label_dict[1]:
                semifactual_latent = (z.to('cuda') + (recent_scalar) * coef).float()
                return temp_latent, semifactual_latent, scalar_new
        return (z.to('cuda') + coef).float(), -1, scalar_new

    def GMM_maker(self, refined_latent_dict, gmm_component):
        gmms = {}
        for gmm_class in tqdm(range(self.num_classes), desc='gmm training'):
            gmms_element = {}
            if refined_latent_dict[gmm_class].shape[0] < self.GMM_component:
                for element in range(self.latent_dim):
                    gmm = GaussianMixture(n_components=gmm_component, random_state=0)
                    gmm = gmm.fit(np.zeros(self.latent_dim).reshape(-1, 1))
                    gmms_element[element] = gmm
                gmms[gmm_class] = gmms_element
                continue
            
            for element in range(self.latent_dim):
                gmm = GaussianMixture(n_components=gmm_component, random_state=0)
                if self.w_space == True:
                    gmm = gmm.fit(refined_latent_dict[gmm_class][:, 0, element].reshape(-1, 1).detach().cpu().numpy())
                elif self.w_space == False:
                    gmm = gmm.fit(refined_latent_dict[gmm_class][:, element].reshape(-1, 1).detach().cpu().numpy())
                gmms_element[element] = gmm
            gmms[gmm_class] = gmms_element
        return gmms

    def center_of_target_maker(self, gmms):
        if self.top_center == True:
            top_logits_dict = {}
            top_latent_dict = {}
            for i in range(self.num_classes):
                top_idx = self.logits_dict[i].max(1).values.sort(descending=True).indices[0]
                top_logits_dict[i] = self.logits_dict[i][top_idx]
                top_latent_dict[i] = self.latent_dict[i][top_idx]
            return top_latent_dict
            
        elif self.top_center == False:
            if self.w_space == True:
                target_mode = torch.zeros([self.num_classes, self.n_latent, self.latent_dim], device='cuda')
            elif self.w_space == False:
                target_mode = torch.zeros([self.num_classes, self.latent_dim], device='cuda')
            
            for from_class in tqdm(range(self.num_classes), desc='center of target maker'):
                for element in range(self.latent_dim):
                    sample_element = gmms[from_class][element].sample(10000)
                    probabiliy_gmm = gmms[from_class][element].score_samples(sample_element[0])
                    max_idx = np.argmax(probabiliy_gmm)
                    if self.w_space == True:
                        temp_mode = torch.tensor(sample_element[0][max_idx]).repeat(self.n_latent)
                        target_mode[from_class][:, element] = temp_mode
                    elif self.w_space == False:
                        target_mode[from_class][element] = torch.tensor(sample_element[0][max_idx])
            return target_mode

    @torch.no_grad()
    def projection_point_finder(self, input_z, coefs, refined_latent_dict, original_class, target_class, target_range, mode=None):
        if mode == 'one':
            coef = coefs
        else:
            coef = coefs[original_class * self.num_classes + target_class]
        # if self.w_space == True:
        #     with torch.no_grad():
        #         z_point = self.generator(input_z, return_only_latent=True)
        # elif self.w_space == False:
        z_point = input_z    
        for modifying_trial in tqdm(range(self.modifying_trial), desc='modifying trial'):
            if self.normal_method == 'nearest' :
                idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[target_class] - z_point)**2)).sum(2)))
                nearest_point = refined_latent_dict[target_class][idx_argmin]
                vector_to_cal = (nearest_point - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'mean':
                # mean_dict = {}
                # for i in range(10):
                #     idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].mean(0))**2)).sum(2)))
                #     mean_dict[i] = refined_latent_dict[i][idx_argmin]
                
                # target_mean = mean_dict[target_class]
                # vector_to_cal = (target_mean - z_point).float()
                # matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                # projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)
                # init_projected_point = projected_point
                
                self.center_of_target = []
                for i in range(self.num_classes):
                    self.center_of_target.append(refined_latent_dict[i].mean(0))
                self.center_of_target = torch.stack(self.center_of_target).to('cuda')
                target_mode = self.center_of_target[target_class]
                
            elif self.normal_method == 'mean_point':
                mean_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].mean(0))**2)).sum(2)))
                    mean_dict[i] = refined_latent_dict[i][idx_argmin]
                target_mean = refined_latent_dict[target_class].mean(0)
                vector_to_cal = (target_mean - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'median':
                self.center_of_target = []
                for i in range(self.num_classes):
                    self.center_of_target.append(refined_latent_dict[i].median(0)[0])
                self.center_of_target = torch.stack(self.center_of_target).to('cuda')
                target_mode = self.center_of_target[target_class]
                # median_dict = {}
                # for i in range(10):
                #     idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].median(0)[0])**2)).sum(2)))
                #     median_dict[i] = refined_latent_dict[i][idx_argmin]
                # target_median = median_dict[target_class]
                # vector_to_cal = (target_median - z_point).float()
                # matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                # projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)
            
            elif self.normal_method == 'median_point':
                median_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].median(0)[0])**2)).sum(2)))
                    median_dict[i] = refined_latent_dict[i][idx_argmin]
                target_median = refined_latent_dict[target_class].median(0)[0]
                vector_to_cal = (target_median - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'mode':
                target_mode = self.center_of_target[target_class]
                
            if self.tabular == False:
                if self.w_space == True:
                    target_mode = target_mode.to('cuda')
                elif self.w_space == False:
                    target_mode = target_mode.reshape(1, self.latent_dim).to('cuda')
            elif self.tabular == True:
                target_mode = target_mode.reshape(1, self.latent_dim).to('cuda')
            vector_to_cal = (target_mode - z_point).float()
            if self.w_space == True:
                matmul_cal = torch.matmul(vector_to_cal[0], coef[0].float())
            elif self.w_space == False:
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.float().squeeze())
            projected_point = z_point + matmul_cal * coef
            init_projected_point = projected_point
            
            if self.tabular == False:
                transformed_sample = self.transform_pil(projected_point.float(), transforms=self.transforms, use_transform=self.use_transform)
                projected_logits = self.classifier(transformed_sample)
            elif self.tabular == True:
                projected_logits = self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                    
            if int(torch.argmax(projected_logits, 1)) in self.class_label_dict.keys():
                projected_pred = int(torch.argmax(projected_logits, 1))
                if projected_pred == self.rev_class_label_dict[1]:
                    print('sucess. found at PP')
                    if projected_point.shape[0] == 1:
                        return projected_point
                    else:
                        return projected_point.unsqueeze(0)
                elif self.use_calibration == False and self.use_perturbing == False:
                    return projected_point
                
            if self.use_calibration == True:
                if self.tabular == False:
                    transformed_sample = self.transform_pil(projected_point.float(), transforms=self.transforms, use_transform=self.use_transform)
                    projected_logits = self.classifier(transformed_sample)
                elif self.tabular == True:
                    projected_logits= self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                projected_pred = int(torch.argmax(projected_logits, 1)) - target_range

                if projected_pred != target_class:
                    if projected_pred == original_class:
                        # for sameclass_trial in tqdm(range(2 * int(1 / self.sub_weight)), desc='same class trial'):
                        current_projected_point = torch.tensor(projected_point, device='cuda')
                        for sameclass_trial in range(2 * int(1 / self.sub_weight)):
                            projected_point = projected_point + coef * self.sub_weight * sameclass_trial
                            if self.tabular == False:
                                transformed_sample = self.transform_pil(projected_point.float(), transforms=self.transforms, use_transform=self.use_transform)
                                projected_logits = self.classifier(transformed_sample)
                            elif self.tabular == True:
                                projected_logits = self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                            projected_pred = torch.argmax(projected_logits, 1) - target_range
                            if projected_pred != original_class:
                                break
                        
                        if projected_pred == target_class:
                            print('sucess')
                            return projected_point.unsqueeze(0)
                        elif projected_pred == original_class:
                            print('fail. it is still original class. the coef is not proper')
                            return current_projected_point
                    
                    similar_class = int(projected_pred)
                    
                    if original_class == similar_class:
                        similar_coef = coefs[original_class * self.num_classes + target_class]
                    else:
                        similar_coef = coefs[original_class * self.num_classes + similar_class]

                    coef = coef - self.sub_weight * similar_coef
                    coef = coef / torch.linalg.norm(coef)
            
            elif self.use_perturbing == True:
                # if self.tabular == False:
                #     transformed_sample = self.transform_pil(projected_point.float(), transforms=self.transforms, use_transform=self.use_transform)
                #     projected_logits = self.classifier(transformed_sample)
                # elif self.tabular == True:
                #     projected_logits = self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                    
                # if int(torch.argmax(projected_logits, 1)) in self.class_label_dict.keys():
                #     projected_pred = self.class_label_dict[int(torch.argmax(projected_logits, 1))]
                #     if projected_pred == target_class:
                #         return projected_point
                positive_point = self.perturber(projected_point=projected_point,
                                                coef=coef, flag='positive', 
                                                target_class=target_class,
                                                iteration=modifying_trial,
                                                target_range=self.class_label_dict)
                negative_point = self.perturber(projected_point=projected_point, 
                                                coef=coef, flag='negative', 
                                                target_class=target_class,
                                                iteration=modifying_trial,
                                                target_range=self.class_label_dict)
                if positive_point is -1 and negative_point is -1 :
                    continue
                elif positive_point is -1 :
                    print(f'success. founded at negative point')
                    return negative_point
                elif negative_point is -1 :
                    print(f'success. founded at positive point')
                    return positive_point
                elif positive_point is not -1 and negative_point is not -1:
                    print(f'success. at both side')
                    diff_posi = ((projected_point - positive_point) ** 2).mean()
                    diff_nega = ((projected_point - negative_point) ** 2).mean()
                    if diff_posi < diff_nega:
                        return positive_point
                    elif diff_nega <= diff_posi:
                        return negative_point
                    
                    # negative_point = self.perturber(projected_point=projected_point, 
                    #                                 coef=coef, flag='negative', 
                    #                                 target_class=target_class,
                    #                                 iteration=modifying_trial)
                    # if negative_point is -1 :
                    #     continue
                    # else:
                        # print(f'success')
                        # return negative_point
                # else:
                #     print(f'fail. `another class` is found{int(torch.argmax(projected_logits, 1))}')
                #     continue
            else:
                if projected_point.shape[0] == 1:
                    return projected_point
                else:
                    return projected_point.unsqueeze(0)
                    
        print(f'fail. the modifying trials over {self.modifying_trial}')
        if init_projected_point.shape[0] == 1:
            return init_projected_point
        else:
            return init_projected_point.unsqueeze(0)
    
    @torch.no_grad()
    def perturber(self, projected_point, coef, flag, target_class, iteration, target_range):
        perturbed_latents = []
        if flag == 'positive':
            start_point = projected_point + coef * self.latent_batch * self.sub_weight * iteration
            end_point = projected_point + coef * self.latent_batch * self.sub_weight * (iteration + 1)
        elif flag == 'negative':
            start_point = projected_point - coef * self.latent_batch * self.sub_weight * iteration
            end_point = projected_point - coef * self.latent_batch * self.sub_weight * (iteration + 1)
        
        if self.w_space == True:
            start_point = start_point[0].unsqueeze(0)
            end_point = end_point[0].unsqueeze(0)
        
        for start_latent, end_latent in zip(start_point, end_point):
            for alpha in torch.linspace(start=0.0, end=1.0, steps=self.latent_batch+1):
                temp_latent = (1.0 - alpha) * start_latent + alpha * end_latent
                perturbed_latents.append(temp_latent)
        perturbed_latents = torch.stack(perturbed_latents).to('cuda').float()
        if self.w_space == True:
            perturbed_latents = perturbed_latents.unsqueeze(1).repeat(1, self.n_latent, 1)
            temp_transformed_img = self.transform_pil(perturbed_latents, transforms=self.transforms, use_transform=self.use_transform)
            projected_pred = torch.argmax(self.classifier(temp_transformed_img), 1)
        elif self.w_space == False:
            projected_pred = torch.argmax(self.classifier(self.generator(perturbed_latents)), 1)
        for i in range(self.latent_batch):
            if projected_pred[i] == self.rev_class_label_dict[1]: # for one algorithm
            # if projected_pred[i] == self.rev_class_label_dict[target_class]: # for k algorithm
                if self.w_space == True:
                    return perturbed_latents[i].unsqueeze(0)
                elif self.w_space == False:
                    return perturbed_latents[i].unsqueeze(0)
        return -1

    
    def SVM_maker(self, z, original_class, target_class, target_range=0):
        print('svm')
        return self.projection_point_finder(input_z=z, coefs=self.SVM_coefs, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class, target_range=target_range)

    def finder_maker(self, z, original_class, target_class, target_range=0, direction=None, mode=None):
        print('finder')
        return self.projection_point_finder(input_z=z, coefs=self.finder_coefs, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class, target_range=target_range)
    
    def finder_maker_one(self, z, original_class, target_class, target_range=0, direction=None, mode=None):
        print('finder')
        if mode == 'one':
            return self.projection_point_finder(input_z=z, coefs=direction, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class, target_range=target_range, mode=mode)
        else:
            return self.projection_point_finder(input_z=z, coefs=direction, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class, target_range=target_range, mode=mode)


class FinderMaker_CelebA_Plausible():
    def __init__(self, projector, classifier, generator, args, CTGAN=None, original_image=None, auged_data=None, tabular=False, transforms=None):
        # self.finder_coef = 1
        self.latent_dim = args.latent_dim
        self.k_count = args.k_count
        self.num_classes = args.num_classes
        # self.projector = projector
        self.classifier = classifier
        self.generator = generator
        self.GMM_component = args.GMM_component
        self.peak_ratio = args.peak_ratio
        self.tabular = tabular
        self.use_transform = args.use_transform
        self.top_center = args.top_center
        self.w_space = args.w_space
        self.n_latent =  args.n_latent
        self.transforms = transforms
        if CTGAN:
            self.CTGAN = CTGAN
        if args.latent_type == 'training':
            self.auged_data = auged_data
            self.original_image = original_image
            self.logits, self.labels = self.logit_label_finder()
            self.refined_logits_dict, self.refined_latent_dict = self.refined_latent_maker()
        elif args.latent_type == 'random':
            self.latent_batch = args.latent_batch
            if args.w_traindata_path is not None:
                self.refined_latent_dict = torch.load(args.w_traindata_path)
            else:
                self.refined_logits_dict, self.refined_latent_dict = self.random_refined_latent_maker(latent_iteration=args.latent_iteration, random_latent_std=args.random_latent_std, latent_batch=self.latent_batch)
        if self.top_center == True:
            self.center_of_target = self.center_of_target_maker(gmms=False)
        elif self.top_center == False:
            self.gmms = self.GMM_maker(refined_latent_dict=self.refined_latent_dict, gmm_component=self.GMM_component)
            self.center_of_target = self.center_of_target_maker(gmms=self.gmms)
        self.SVM_coefs = self.SVM_coef_maker()
        # self.finder_coefs = self.finder_coef_maker()
        self.normal_method = args.normal_method
        self.path = args.path
        self.sub_weight = args.sub_weight      
        self.input_latent_path = args.input_latent_path
        self.random_latent_std = args.random_latent_std
        self.modifying_trial = args.modifying_trial
        self.use_calibration = args.use_calibration
        self.use_perturbing = args.use_perturbing

    @torch.no_grad()
    def transform_pil(self, latent, transforms, use_transform=True):
        temp_batch = len(latent)
        rtn = []
        z = latent
        if latent.shape[0] >= 2:
            temp_generated = self.generator(latent=latent.unsqueeze(0))
            return temp_generated
        temp_generated, w = self.generator(z.to('cuda'))
        if use_transform == True:
            for i in range(temp_batch):
                pil_image = tensor_to_pil_transform(torchvision.utils.make_grid(temp_generated[i], normalize=True))
                transformed_sample = transforms(pil_image)
                rtn.append(transformed_sample)
            rtn = torch.stack(rtn)
            return rtn.to('cuda')
        elif use_transform == False:
            return temp_generated, w
        
    def refined_latent_maker(self):
        logits_dict = {}
        latent_dict = {}
        with torch.no_grad():
            logits, _ = self.classifier(self.original_image / 255 * 2 - 1)
        for i in range(self.num_classes):
            logits_dict[i] = logits[torch.where(self.labels == i)]
            latent_dict[i] = self.auged_data[torch.where(self.labels == i)]

        refined_logits_dict = {}
        refined_latent_dict = {}
        for i in range(self.num_classes):
            top_idx = logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(logits_dict[i]) * self.peak_ratio)]
            refined_logits_dict[i] = logits_dict[i][top_idx]
            refined_latent_dict[i] = latent_dict[i][top_idx]
        return refined_logits_dict, refined_latent_dict

    def generate_activate_transform(self, input, CTGAN, only_activate=False):
        gen_data = CTGAN._generator(input)
        activated_data = CTGAN._apply_activate(gen_data)
        if only_activate==True:
            return activated_data
        elif only_activate==False:
            inv_transformed_data = CTGAN._transformer.inverse_transform(activated_data.detach().cpu().numpy())
            return inv_transformed_data, activated_data
        

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
            

        for iteration in tqdm(range(latent_iteration), desc='latent generating'):
            if self.tabular==False:
                if self.w_space == True:
                    z = random_latent_std * torch.randn([latent_batch, 1, self.latent_dim]).to('cuda')
                    z = z.repeat(1, self.n_latent, 1)
                else:
                    z = random_latent_std * torch.randn([latent_batch, self.latent_dim]).to('cuda')
            elif self.tabular==True:
                z = random_latent_std * torch.randn([latent_batch, self.latent_dim]).to('cuda')
            if self.tabular==False:
                if self.w_space == True:
                    transformed_sample = self.transform_pil(z, transforms=self.transforms, use_transform=self.use_transform)
                elif self.w_space == False:
                    transformed_sample = self.generator(z)[0]
                logits_from_z.append(self.classifier(transformed_sample).to('cpu'))
            elif self.tabular==True:
                activated_data = self.generate_activate_transform(z.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
                logits_from_z.append(self.classifier(activated_data).to('cpu'))
            if self.w_space == True:
                z_sum[iteration * latent_batch : (iteration + 1) * latent_batch] = w
            elif self.w_space == False:
                z_sum[iteration * latent_batch : (iteration + 1) * latent_batch] = z


        logits_from_z = torch.cat(logits_from_z)
        pred = torch.argmax(logits_from_z, 1)
        for i in range(self.num_classes):
            self.logits_dict[i] = logits_from_z[torch.where(pred == i)]
            self.latent_dict[i] = z_sum[torch.where(pred == i)]
        
        refined_logits_dict = {}
        refined_latent_dict = {}
        for i in range(self.num_classes):
            top_idx = self.logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(self.logits_dict[i]) * self.peak_ratio)]
            refined_logits_dict[i] = self.logits_dict[i][top_idx]
            refined_latent_dict[i] = self.latent_dict[i][top_idx]
        return refined_logits_dict, refined_latent_dict
    
    def SVM_coef_maker(self):
        if self.tabular==False:
            if self.w_space == True:
                SVM_coefs = torch.zeros([self.k_count, 14, self.latent_dim])
            elif self.w_space == False:
                SVM_coefs = torch.zeros([self.k_count, 1, self.latent_dim])
        if self.tabular==True:
            SVM_coefs = torch.zeros([self.k_count, 1, self.latent_dim])
        
        SVM_coefs = []
        for from_class in tqdm(range(self.num_classes), desc='SVM coef maker'):
            for to_class in range(self.num_classes):
                svc_model = SVC(C=1.0, kernel='linear')
                if self.w_space == True:
                    sample_origin_target_data = torch.cat([self.refined_latent_dict[from_class][:, 0], self.refined_latent_dict[to_class][:, 0]])
                elif self.w_space == False:
                    sample_origin_target_data = torch.cat([self.refined_latent_dict[from_class], self.refined_latent_dict[to_class]])
                # if len(self.refined_latent_dict[from_class]) == 0 or len(self.refined_latent_dict[to_class]) == 0:
                #     temp_direction = torch.zeros(1, self.latent_dim)
                #     SVM_coefs.append(temp_direction.to('cuda'))
                #     continue
                sample_origin_target_label = torch.cat([torch.zeros(len(self.refined_latent_dict[from_class])), torch.ones(len(self.refined_latent_dict[to_class]))])
                sample_origin_target_data = sample_origin_target_data.detach().cpu().numpy()
                sample_origin_target_label = sample_origin_target_label.detach().cpu().numpy()

                svc_model.fit(sample_origin_target_data.squeeze(), sample_origin_target_label)
                temp_coef = svc_model.coef_ / np.linalg.norm(svc_model.coef_)
                temp_coef = temp_coef.astype(float)
                if self.tabular==False:
                    if self.w_space == True:
                        temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)
                        temp_coef = temp_coef.repeat(self.n_latent, 1)
                    elif self.w_space == False:
                        temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)
                elif self.tabular==True:
                    temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)

                SVM_coefs.append(temp_coef)
                # SVM_coefs[torch.stack([from_class, to_class])] = temp_coef
        SVM_coefs = torch.stack(SVM_coefs)
        return SVM_coefs.to('cuda')

    def finder_coef_maker(self):
        if self.tabular==False:
            finder_coefs = torch.zeros([self.k_count, 1, self.latent_dim], device='cuda')
        elif self.tabular==True:
            finder_coefs = torch.zeros([self.k_count, 1, self.latent_dim], device='cuda')
            
        finder_coefs = []
        for idx in range(self.k_count):
            sample = torch.zeros(self.k_count, device='cuda')
            sample[idx] = 1
            sample = sample.to('cuda')
            with torch.no_grad():
                if self.tabular==False:
                    temp_finder_coef = self.projector(sample).reshape(1, self.latent_dim)
                elif self.tabular==True:
                    temp_finder_coef = self.projector(sample).reshape(1, self.latent_dim)
                temp_finder_coef = temp_finder_coef / torch.linalg.norm(temp_finder_coef)

                finder_coefs.append(temp_finder_coef)
        finder_coefs = torch.stack(finder_coefs)
        if self.w_space == True:
            return finder_coefs.repeat(1, self.n_latent, 1).to('cuda')
        elif self.w_space == False:
            return finder_coefs.to('cuda')

    def logit_label_finder(self):
        input_image = self.original_image.to('cpu')
        cpu_classifier = self.classifier.to('cpu')
        logits = cpu_classifier(input_image / 255 * 2 - 1)[0]
        softmax_func = torch.nn.Softmax(dim=1)
        softmax_result = softmax_func(logits)
        labels = softmax_result.argmax(1)
        logits = logits.to('cuda')
        labels = labels.to('cuda')
        return logits, labels

    def direction_maker(self, z, original_class, target_class, coef_name):
        if coef_name == 'svm':
            coef = self.SVM_coefs[original_class * self.num_classes + target_class]
        elif coef_name == 'finder':
            coef = self.finder_coefs[original_class * self.num_classes + target_class]
        scalar=0
        for trial in range(1):
            z_direction, z_semifactual, scalar = self.latent_finder(latent=z, 
                                                    coef=coef, 
                                                    target_class=target_class, 
                                                    recent_scalar=scalar, 
                                                    scalar_start=0, 
                                                    scalar_end=5, 
                                                    num=500, 
                                                    debug_mode=False)

            with torch.no_grad():
                if self.tabular==False:
                    transformed_sample, w = self.transform_pil(z_direction, transforms=self.transforms, use_transform=self.use_transform)
                    temp_logits = self.classifier(transformed_sample)
                elif self.tabular==True:
                    temp_activated = self.generate_activate_transform(z_direction.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
                    temp_logits = self.classifier(temp_activated)
                temp_pred = int(torch.argmax(temp_logits))
            if temp_pred == target_class:
                return z_direction, z_semifactual
            
        print('not found in direction')
        return z_direction

    @torch.no_grad()
    def latent_finder(self, latent, coef, target_class, recent_scalar, scalar_start=0, scalar_end=10, num=1000, debug_mode=False):
        z = latent
        for scalar in tqdm(np.linspace(scalar_start, scalar_end, num), desc='direction_maker'):
            if scalar == 0:
                continue
            scalar = torch.tensor(scalar, dtype=float).to('cuda')
            scalar_new = scalar + recent_scalar
            # print(f'scalar:{scalar_new}')
            if self.tabular == False:
                if self.w_space == True:
                    temp_latent = (z.to('cuda') + (scalar_new) * coef).float()
                elif self.w_space == False:
                    temp_latent = (z.to('cuda') + (scalar_new) * coef.reshape(1, self.latent_dim)).float()
                transformed_sample, w = self.transform_pil(temp_latent, transforms=self.transforms, use_transform=self.use_transform)
                logits = self.classifier(transformed_sample)
            elif self.tabular == True:
                temp_latent = (z.to('cuda') + (scalar_new) * coef.reshape(1, self.latent_dim)).float()
                temp_activated = self.generate_activate_transform(temp_latent, self.CTGAN, only_activate=True)
                logits = self.classifier(temp_activated)
            
            pred = int(torch.argmax(logits, 1))

            if pred == target_class:
                semifactual_latent = (z.to('cuda') + (recent_scalar) * coef).float()
                return temp_latent, semifactual_latent, scalar_new
        return temp_latent, -1, scalar_new

    def GMM_maker(self, refined_latent_dict, gmm_component):
        gmms = {}
        for gmm_class in tqdm(range(self.num_classes), desc='gmm training'):
            gmms_element = {}
            if refined_latent_dict[gmm_class].shape[0] < self.GMM_component:
                for element in range(self.latent_dim):
                    gmm = GaussianMixture(n_components=gmm_component, random_state=0)
                    gmm = gmm.fit(np.zeros(self.latent_dim).reshape(-1, 1))
                    gmms_element[element] = gmm
                gmms[gmm_class] = gmms_element
                continue
            
            for element in range(self.latent_dim):
                gmm = GaussianMixture(n_components=gmm_component, random_state=0)
                if self.w_space == True:
                    gmm = gmm.fit(refined_latent_dict[gmm_class][:, 0, element].reshape(-1, 1).detach().cpu().numpy())
                elif self.w_space == False:
                    gmm = gmm.fit(refined_latent_dict[gmm_class][:, element].reshape(-1, 1).detach().cpu().numpy())
                gmms_element[element] = gmm
            gmms[gmm_class] = gmms_element
        return gmms

    def center_of_target_maker(self, gmms):
        if self.top_center == True:
            top_logits_dict = {}
            top_latent_dict = {}
            for i in range(self.num_classes):
                top_idx = self.logits_dict[i].max(1).values.sort(descending=True).indices[0]
                top_logits_dict[i] = self.logits_dict[i][top_idx]
                top_latent_dict[i] = self.latent_dict[i][top_idx]
            return top_latent_dict
            
        elif self.top_center == False:
            if self.w_space == True:
                target_mode = torch.zeros([self.num_classes, self.n_latent, self.latent_dim], device='cuda')
            elif self.w_space == False:
                target_mode = torch.zeros([self.num_classes, self.latent_dim], device='cuda')
            
            for from_class in tqdm(range(self.num_classes), desc='center of target maker'):
                for element in range(self.latent_dim):
                    sample_element = gmms[from_class][element].sample(10000)
                    probabiliy_gmm = gmms[from_class][element].score_samples(sample_element[0])
                    max_idx = np.argmax(probabiliy_gmm)
                    if self.w_space == True:
                        temp_mode = torch.tensor(sample_element[0][max_idx]).repeat(self.n_latent)
                        target_mode[from_class][:, element] = temp_mode
                    elif self.w_space == False:
                        target_mode[from_class][element] = torch.tensor(sample_element[0][max_idx])
            return target_mode

    @torch.no_grad()
    def projection_point_finder(self, input_z, coefs, refined_latent_dict, original_class, target_class, mode=None):
        if mode == 'one':
            coef = coefs
        else:
            coef = coefs[original_class * self.num_classes + target_class]
        # if self.w_space == True:
        #     with torch.no_grad():
        #         z_point = self.generator(input_z, return_only_latent=True)
        # elif self.w_space == False:
        z_point = input_z    
        for modifying_trial in tqdm(range(self.modifying_trial), desc='modifying trial'):
            if self.normal_method == 'nearest' :
                idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[target_class] - z_point)**2)).sum(2)))
                nearest_point = refined_latent_dict[target_class][idx_argmin]
                vector_to_cal = (nearest_point - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'mean':
                mean_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].mean(0))**2)).sum(2)))
                    mean_dict[i] = refined_latent_dict[i][idx_argmin]
                target_mean = mean_dict[target_class]
                vector_to_cal = (target_mean - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)
            
            elif self.normal_method == 'mean_point':
                mean_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].mean(0))**2)).sum(2)))
                    mean_dict[i] = refined_latent_dict[i][idx_argmin]
                target_mean = refined_latent_dict[target_class].mean(0)
                vector_to_cal = (target_mean - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'median':
                median_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].median(0)[0])**2)).sum(2)))
                    median_dict[i] = refined_latent_dict[i][idx_argmin]
                target_median = median_dict[target_class]
                vector_to_cal = (target_median - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)
            
            elif self.normal_method == 'median_point':
                median_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].median(0)[0])**2)).sum(2)))
                    median_dict[i] = refined_latent_dict[i][idx_argmin]
                target_median = refined_latent_dict[target_class].median(0)[0]
                vector_to_cal = (target_median - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'mode':
                target_mode = self.center_of_target[target_class]
                if self.tabular == False:
                    if self.w_space == True:
                        target_mode = target_mode.to('cuda')
                    elif self.w_space == False:
                        target_mode = target_mode.reshape(1, self.latent_dim).to('cuda')
                elif self.tabular == True:
                    target_mode = target_mode.reshape(1, self.latent_dim).to('cuda')
                vector_to_cal = (target_mode - z_point).float()
                if self.w_space == True:
                    matmul_cal = torch.matmul(vector_to_cal[0], coef[0].float())
                elif self.w_space == False:
                    matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.float().squeeze())
                projected_point = z_point + matmul_cal * coef
                
            if self.use_calibration == True:
                if self.tabular == False:
                    
                    transformed_sample = self.transform_pil(projected_point.float(), transforms=self.transforms, use_transform=self.use_transform)
                    projected_logits = self.classifier(transformed_sample)
                elif self.tabular == True:
                    projected_logits= self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                projected_pred = int(torch.argmax(projected_logits, 1))

                if projected_pred != target_class:
                    if projected_pred == original_class:
                        # for sameclass_trial in tqdm(range(2 * int(1 / self.sub_weight)), desc='same class trial'):
                        current_projected_point = torch.tensor(projected_point, device='cuda')
                        for sameclass_trial in range(2 * int(1 / self.sub_weight)):
                            projected_point = projected_point + coef * self.sub_weight * sameclass_trial
                            if self.tabular == False:
                                transformed_sample = self.transform_pil(projected_point.float(), transforms=self.transforms, use_transform=self.use_transform)
                                projected_logits = self.classifier(transformed_sample)
                            elif self.tabular == True:
                                projected_logits = self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                            projected_pred = torch.argmax(projected_logits, 1)
                            if projected_pred != original_class:
                                break
                        
                        if projected_pred == target_class:
                            print('sucess')
                            return projected_point
                        elif projected_pred == original_class:
                            print('fail. it is still original class. the coef is not proper')
                            return current_projected_point
                    
                    similar_class = int(projected_pred)
                    
                    if original_class == similar_class:
                        similar_coef = coefs[original_class * self.num_classes + target_class]
                    else:
                        similar_coef = coefs[original_class * self.num_classes + similar_class]

                    coef = coef - self.sub_weight * similar_coef
                    coef = coef / torch.linalg.norm(coef)
                    
            elif self.use_perturbing == True:
                if self.tabular == False:
                    transformed_sample, w = self.transform_pil(projected_point.float(), transforms=self.transforms, use_transform=self.use_transform)
                    projected_logits = self.classifier(transformed_sample)
                elif self.tabular == True:
                    projected_logits= self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                projected_pred = int(torch.argmax(projected_logits, 1))
                
                if projected_pred == target_class:
                    return projected_point

                elif projected_pred != target_class:
                    positive_point = self.perturber(projected_point=projected_point,
                                                    coef=coef, flag='positive', 
                                                    target_class=target_class,
                                                    iteration=modifying_trial)
                    negative_point = self.perturber(projected_point=projected_point, 
                                                    coef=coef, flag='negative', 
                                                    target_class=target_class,
                                                    iteration=modifying_trial)
                    if positive_point is -1 and negative_point is -1 :
                        continue
                    elif positive_point is -1 :
                        return negative_point
                    elif negative_point is -1 :
                        return positive_point
                    elif positive_point is not -1 and negative_point is not -1:
                        print(f'success')
                        diff_posi = ((projected_point - positive_point) ** 2).mean()
                        diff_nega = ((projected_point - negative_point) ** 2).mean()
                        if diff_posi < diff_nega:
                            return positive_point
                        elif diff_nega <= diff_posi:
                            return negative_point
                        
                    # negative_point = self.perturber(projected_point=projected_point, 
                    #                                 coef=coef, flag='negative', 
                    #                                 target_class=target_class,
                    #                                 iteration=modifying_trial)
                    # if negative_point is -1 :
                    #     continue
                    # else:
                        # print(f'success')
                        # return negative_point
                        
            else:
                return projected_point
                    
        print(f'fail. the modifying trials over {self.modifying_trial}')
        return projected_point
    
    @torch.no_grad()
    def perturber(self, projected_point, coef, flag, target_class, iteration):
        perturbed_latents = []
        
        if flag == 'positive':
            start_point = projected_point + coef * self.latent_batch * self.sub_weight * iteration
            end_point = projected_point + coef * self.latent_batch * self.sub_weight * (iteration + 1)
        elif flag == 'negative':
            start_point = projected_point - coef * self.latent_batch * self.sub_weight * iteration
            end_point = projected_point - coef * self.latent_batch * self.sub_weight * (iteration + 1)
        
        if self.w_space == True:
            start_point = start_point[0].unsqueeze(0)
            end_point = end_point[0].unsqueeze(0)
        
        for start_latent, end_latent in zip(start_point, end_point):
            for alpha in torch.linspace(start=0.0, end=1.0, steps=self.latent_batch+1):
                temp_latent = alpha * start_latent + (1.0 - alpha) * end_latent
                perturbed_latents.append(temp_latent)
        perturbed_latents = torch.stack(perturbed_latents).to('cuda').float()
        if self.w_space == True:
            perturbed_latents = perturbed_latents.unsqueeze(1).repeat(1, self.n_latent, 1)
            projected_pred = torch.argmax(self.classifier(self.generator(latent = perturbed_latents)[0]), 1)
        elif self.w_space == False:
            projected_pred = torch.argmax(self.classifier(self.generator(perturbed_latents)[0]), 1)
        for i in range(self.latent_batch):
            if projected_pred[i] == target_class:
                return perturbed_latents[i]
        return -1

    
    def SVM_maker(self, z, original_class, target_class):
        return self.projection_point_finder(input_z=z, coefs=self.SVM_coefs, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class)

    def finder_maker(self, z, original_class, target_class):
        return self.projection_point_finder(input_z=z, coefs=self.finder_coefs, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class)



class FinderMaker_CelebA():
    def __init__(self, projector, classifier, generator, args, CTGAN=None, original_image=None, auged_data=None, tabular=False, transforms=None):
        self.finder_coef = 1
        self.latent_dim = args.latent_dim
        self.k_count = args.k_count
        self.num_classes = args.num_classes
        self.projector = projector
        self.classifier = classifier
        self.generator = generator
        self.GMM_component = args.GMM_component
        self.peak_ratio = args.peak_ratio
        self.tabular = tabular
        self.all_classes = self.class_maker(args.k_count)
        if transforms is not None:
            self.transforms = transforms
        if CTGAN:
            self.CTGAN = CTGAN
        if args.latent_type == 'training':
            self.auged_data = auged_data
            self.original_image = original_image
            self.logits, self.labels = self.logit_label_finder()
            self.refined_logits_dict, self.refined_latent_dict = self.refined_latent_maker()
        elif args.latent_type == 'random':
            self.refined_logits_dict, self.refined_latent_dict = self.random_refined_latent_maker(latent_iteration=args.latent_iteration, random_latent_std=args.random_latent_std, latent_batch=args.latent_batch)
        self.gmms = self.GMM_maker(refined_latent_dict=self.refined_latent_dict, gmm_component=self.GMM_component)
        self.center_of_target = self.center_of_target_maker(gmms=self.gmms)
        self.SVM_coefs = self.SVM_coef_maker()
        self.finder_coefs = self.finder_coef_maker()
        self.normal_method = args.normal_method
        self.path = args.path
        self.sub_weight = args.sub_weight      
        self.input_latent_path = args.input_latent_path
        self.random_latent_std = args.random_latent_std
        self.modifying_trial = args.modifying_trial
        self.sigmoid = nn.Sigmoid() # adding needed
        

    def transform_pil(self, latent, transforms):
        temp_batch = len(latent)
        rtn = []
        z = latent
        temp_generated = self.generator(z.to('cuda'))[0]
        for i in range(temp_batch):
            pil_image = tensor_to_pil_transform(torchvision.utils.make_grid(temp_generated[i], normalize=True))
            transformed_sample = transforms(pil_image)
            rtn.append(transformed_sample)
        rtn = torch.stack(rtn)
        return rtn.to('cuda')
        
    def refined_latent_maker(self):
        logits_dict = {}
        latent_dict = {}
        with torch.no_grad():
            logits, _ = self.classifier(self.original_image / 255 * 2 - 1)
        for i in range(self.num_classes):
            logits_dict[i] = logits[torch.where(self.labels == i)]
            latent_dict[i] = self.auged_data[torch.where(self.labels == i)]

        refined_logits_dict = {}
        refined_latent_dict = {}
        for i in range(self.num_classes):
            top_idx = logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(logits_dict[i]) * self.peak_ratio)]
            refined_logits_dict[i] = logits_dict[i][top_idx]
            refined_latent_dict[i] = latent_dict[i][top_idx]
        return refined_logits_dict, refined_latent_dict

    def generate_activate_transform(self, input, CTGAN, only_activate=False):
        gen_data = CTGAN._generator(input)
        activated_data = CTGAN._apply_activate(gen_data)
        if only_activate==True:
            return activated_data
        elif only_activate==False:
            inv_transformed_data = CTGAN._transformer.inverse_transform(activated_data.detach().cpu().numpy())
            return inv_transformed_data, activated_data
        

    @torch.no_grad()
    def random_refined_latent_maker(self, latent_iteration, random_latent_std=None, latent_batch=None):
        if random_latent_std is None:
            random_latent_std = self.random_latent_std
        logits_dict = {}
        latent_dict = {}

        if self.tabular==False:
            z_sum = torch.zeros([latent_batch * latent_iteration, self.latent_dim])
        elif self.tabular==True:
            z_sum = torch.zeros([latent_batch * latent_iteration, self.latent_dim])

        logits_from_z = []
        
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
            

        for iter in tqdm(range(latent_iteration), desc='latent generating'):
            if self.tabular==False:
                z = random_latent_std * torch.randn([latent_batch, self.latent_dim])
            elif self.tabular==True:
                z = random_latent_std * torch.randn([latent_batch, self.latent_dim])
            z_sum[iter * latent_batch : (iter + 1) * latent_batch] = z
            if self.tabular==False:
                transformed_sample = self.transform_pil(z, transforms=self.transforms)
                logits_from_z.append(self.classifier(transformed_sample).to('cpu'))
            elif self.tabular==True:
                activated_data = self.generate_activate_transform(z.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
                logits_from_z.append(self.classifier(activated_data).to('cpu'))


        # img_ = []
        # logits_from_z = []
        # for iter in range(latent_iteration):
        #     if self.tabular==False:
        #         z = random_latent_std * torch.randn([latent_batch, self.latent_dim])
        #     elif self.tabular==True:
        #         z = random_latent_std * torch.randn([latent_batch, self.latent_dim])
        #     z_sum[iter * latent_batch : (iter + 1) * latent_batch] = z
        #     if self.tabular==False:
        #         logits_from_z.append(self.classifier(self.generator(z.to('cuda'))[0]).to('cpu'))
        #         img_.append(self.generator(z.to('cuda'))[0].to('cpu'))
        #     elif self.tabular==True:
        #         activated_data = self.generate_activate_transform(z.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
        #         logits_from_z.append(self.classifier(activated_data).to('cpu'))
        # img_ = torch.cat(img_)
    
        # pil_image = PIL.Image.fromarray(np.array(torchvision.utils.make_grid(img_, normalize=True).permute(1, 2, 0)*255, dtype=np.uint8))
        # pil_image.save(f'./data/CelebA-HQ/temp_image/test.png')

        sigmoid = nn.Sigmoid()
        logits_from_z = torch.cat(logits_from_z)
        pred = torch.round(sigmoid(logits_from_z))
        for each_class in self.all_classes:
            temp_list = []
            for each_pred in range(pred.shape[0]):
                temp_list.append(torch.equal(pred[each_pred], each_class))
            logits_dict[each_class] = logits_from_z[temp_list]
            latent_dict[each_class] = z_sum[temp_list]

        # refined_logits_dict = {}
        # refined_latent_dict = {}
        # for i in range(self.num_classes):
        #     top_idx = logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(logits_dict[i]) * self.peak_ratio)]
        #     refined_logits_dict[i] = logits_dict[i][top_idx]
        #     refined_latent_dict[i] = latent_dict[i][top_idx]
        return logits_dict, latent_dict
    
    def class_maker(self, k_count):
        rtn = torch.zeros(2**k_count, k_count)
        for i in range(2**k_count):
            s = bin(i)[2:]
            s = '0' * (k_count - len(s)) + s
            rtn[i] = torch.tensor(list(map(float, list(s))))
        return rtn

    def SVM_coef_maker(self):
        # if self.tabular==False:
        #     SVM_coefs = torch.zeros([self.k_count, 1, self.latent_dim])
        # if self.tabular==True:
        #     SVM_coefs = torch.zeros([self.k_count, 1, self.latent_dim])
            
        # all_classes = self.class_maker(self.k_count)
        
        SVM_coefs = []
        for from_class in tqdm(self.refined_latent_dict.keys(), desc='SVM coef maker'):
            for to_class in self.refined_latent_dict.keys():
                svc_model = SVC(C=1.0, kernel='linear')
                sample_origin_target_data = torch.cat([self.refined_latent_dict[from_class], self.refined_latent_dict[to_class]])
                if len(self.refined_latent_dict[from_class]) == 0 or len(self.refined_latent_dict[to_class]) == 0:
                    temp_direction = torch.zeros(1, self.latent_dim)
                    SVM_coefs.append(temp_direction.to('cuda'))
                    continue
                sample_origin_target_label = torch.cat([torch.zeros(len(self.refined_latent_dict[from_class])), torch.ones(len(self.refined_latent_dict[to_class]))])
                sample_origin_target_data = sample_origin_target_data.detach().cpu().numpy()
                sample_origin_target_label = sample_origin_target_label.detach().cpu().numpy()

                svc_model.fit(sample_origin_target_data.squeeze(), sample_origin_target_label)
                temp_coef = svc_model.coef_ / np.linalg.norm(svc_model.coef_)
                temp_coef = temp_coef.astype(float)
                if self.tabular==False:
                    temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)
                elif self.tabular==True:
                    temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)

                SVM_coefs.append(temp_coef)
                # SVM_coefs[torch.stack([from_class, to_class])] = temp_coef
        SVM_coefs = torch.stack(SVM_coefs)
        return SVM_coefs.to('cuda')

    def finder_coef_maker(self):
        if self.tabular==False:
            finder_coefs = torch.zeros([self.k_count, 1, self.latent_dim], device='cuda')
        elif self.tabular==True:
            finder_coefs = torch.zeros([self.k_count, 1, self.latent_dim], device='cuda')
            
        finder_coefs = []
        for sample in self.refined_latent_dict.keys():
            # sample = torch.zeros(self.k_count, device='cuda')
            # sample[idx] = 1
            sample = sample.to('cuda')
            with torch.no_grad():
                if self.tabular==False:
                    temp_finder_coef = self.projector(sample).reshape(1, self.latent_dim)
                elif self.tabular==True:
                    temp_finder_coef = self.projector(sample).reshape(1, self.latent_dim)
                temp_finder_coef = temp_finder_coef / torch.linalg.norm(temp_finder_coef)

                finder_coefs.append(temp_finder_coef)
        finder_coefs = torch.stack(finder_coefs)
        return finder_coefs.to('cuda')

    def logit_label_finder(self):
        input_image = self.original_image.to('cpu')
        cpu_classifier = self.classifier.to('cpu')
        logits = cpu_classifier(input_image / 255 * 2 - 1)[0]
        softmax_func = torch.nn.Softmax(dim=1)
        softmax_result = softmax_func(logits)
        labels = softmax_result.argmax(1)
        logits = logits.to('cuda')
        labels = labels.to('cuda')
        return logits, labels

    def direction_maker(self, z, original_class, target_class, coef_name):
        if coef_name == 'svm':
            coef = self.SVM_coefs[original_class * self.num_classes + target_class]
        elif coef_name == 'finder':
            coef = self.finder_coefs[original_class * self.num_classes + target_class]
        scalar=0
        for trial in tqdm(range(1000), desc='direction_maker'):
            z_direction, scalar = self.latent_finder(latent=z, 
                                                    coef=coef, 
                                                    target_class=target_class, 
                                                    recent_scalar=scalar, 
                                                    scalar_start=0, 
                                                    scalar_end=10, 
                                                    num=1000, 
                                                    debug_mode=False)

            with torch.no_grad():
                if self.tabular==False:
                    transformed_sample = self.transform_pil(z_direction, transforms=self.transforms)
                    temp_logits = self.classifier(transformed_sample)
                elif self.tabular==True:
                    temp_activated = self.generate_activate_transform(z_direction.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
                    temp_logits = self.classifier(temp_activated)
                temp_pred = int(torch.argmax(temp_logits))
            if temp_pred == target_class:
                return z_direction
        print('not found in direction')

    @torch.no_grad()
    def latent_finder(self, latent, coef, target_class, recent_scalar, scalar_start=0, scalar_end=100, num=100000, debug_mode=False):
        z = latent
        for scalar in np.linspace(scalar_start, scalar_end, num):
            if scalar == 0:
                continue
            scalar = torch.tensor(scalar, dtype=float).to('cuda')
            scalar_new = scalar + recent_scalar
            # print(f'scalar:{scalar_new}')
            if self.tabular == False:
                temp_latent = (z.to('cuda') + (scalar_new) * coef.reshape(1, self.latent_dim)).float()
                transformed_sample = self.transform_pil(temp_latent, transforms=self.transforms)
                logits = self.classifier(transformed_sample)
            elif self.tabular == True:
                temp_latent = (z.to('cuda') + (scalar_new) * coef.reshape(1, self.latent_dim)).float()
                temp_activated = self.generate_activate_transform(temp_latent, self.CTGAN, only_activate=True)
                logits = self.classifier(temp_activated)
            
            pred = int(torch.argmax(logits, 1))
            # if debug_mode == True and self.tabular==False:
            #     plt.imshow(temp_generated_image.squeeze().detach().cpu())
            #     plt.title(f'pred_label:{pred}, scalar:{scalar_new:.1} for directing coef to test dataset')
            #     plt.show()
            if pred == target_class:
                return temp_latent, scalar_new
        return temp_latent, scalar_new

    def GMM_maker(self, refined_latent_dict, gmm_component):
        gmms = {}
        for gmm_class in tqdm(refined_latent_dict.keys(), desc='gmm training'):
            gmms_element = {}
            if refined_latent_dict[gmm_class].shape[0] < self.GMM_component:
                for element in range(self.latent_dim):
                    gmm = GaussianMixture(n_components=gmm_component, random_state=0)
                    gmm = gmm.fit(np.zeros(self.latent_dim).reshape(-1, 1))
                    gmms_element[element] = gmm
                gmms[gmm_class] = gmms_element
                continue
            
            for element in range(self.latent_dim):
                gmm = GaussianMixture(n_components=gmm_component, random_state=0)
                gmm = gmm.fit(refined_latent_dict[gmm_class][:, element].reshape(-1, 1).detach().cpu().numpy())
                gmms_element[element] = gmm
            gmms[gmm_class] = gmms_element
        return gmms

    def center_of_target_maker(self, gmms):
        target_mode = torch.zeros([self.num_classes, self.latent_dim], device='cuda')
        for from_class in tqdm(gmms.keys(), desc='center of target maker'):
            for element in range(self.latent_dim):
                # target_mode[from_class] = torch.zeros(self.latent_dim, device='cuda')
                sample_element = gmms[from_class][element].sample(10000)
                probabiliy_gmm = gmms[from_class][element].score_samples(sample_element[0])
                max_idx = np.argmax(probabiliy_gmm)
                target_mode[from_class][element] = torch.tensor(sample_element[0][max_idx])
        return target_mode

    def class_direction_finder(total_class, dest_class):
        for i in range(len(total_class)):
            if torch.equal(total_class[i], dest_class):
                return i
    
    @torch.no_grad()
    def projection_point_finder(self, input_z, coefs, refined_latent_dict, original_class, target_class, coef_idx):
        
        matching_class = []
        for from_class in self.all_classes:
            for to_class in self.all_classes:
                matching_class.append(torch.stack([from_class, to_class]))
        matching_class = torch.stack(matching_class)

        coef = coefs[coef_idx]
        z_point = input_z
        for modifying_trial in tqdm(range(self.modifying_trial), desc='modifying trial'):
            if self.normal_method == 'nearest' :
                idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[target_class] - z_point)**2)).sum(2)))
                nearest_point = refined_latent_dict[target_class][idx_argmin]
                vector_to_cal = (nearest_point - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'mean':
                mean_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].mean(0))**2)).sum(2)))
                    mean_dict[i] = refined_latent_dict[i][idx_argmin]
                target_mean = mean_dict[target_class]
                vector_to_cal = (target_mean - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)
            
            elif self.normal_method == 'mean_point':
                mean_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].mean(0))**2)).sum(2)))
                    mean_dict[i] = refined_latent_dict[i][idx_argmin]
                target_mean = refined_latent_dict[target_class].mean(0)
                vector_to_cal = (target_mean - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'median':
                median_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].median(0)[0])**2)).sum(2)))
                    median_dict[i] = refined_latent_dict[i][idx_argmin]
                target_median = median_dict[target_class]
                vector_to_cal = (target_median - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)
            
            elif self.normal_method == 'median_point':
                median_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].median(0)[0])**2)).sum(2)))
                    median_dict[i] = refined_latent_dict[i][idx_argmin]
                target_median = refined_latent_dict[target_class].median(0)[0]
                vector_to_cal = (target_median - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim)

            elif self.normal_method == 'mode':
                for key in self.center_of_target.keys():
                    if torch.equal(key.to('cuda'), target_class.squeeze()):
                        break
                target_mode = self.center_of_target[key]
                if self.tabular == False:
                    target_mode = target_mode.reshape(1, self.latent_dim).to('cuda')
                elif self.tabular == True:
                    target_mode = target_mode.reshape(1, self.latent_dim).to('cuda')
                vector_to_cal = (target_mode - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.float().squeeze())
                projected_point = z_point + matmul_cal * coef

            if self.tabular == False:
                transformed_sample = self.transform_pil(projected_point.float(), transforms=self.transforms)
                projected_logits = self.classifier(transformed_sample)
            elif self.tabular == True:
                projected_logits= self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
            self.sigmoid = torch.nn.Sigmoid()
            projected_pred = torch.round(self.sigmoid(projected_logits))

            if torch.equal(projected_pred, target_class) == False:
                if torch.equal(projected_pred, original_class):
                    for sameclass_trial in tqdm(range(2 * int(1 / self.sub_weight)), desc='same class trial'):
                    # for sameclass_trial in range(2 * int(1 / self.sub_weight)):
                        projected_point = projected_point + coef * self.sub_weight * sameclass_trial
                        if self.tabular == False:
                            transformed_sample = self.transform_pil(projected_point.float(), transforms=self.transforms)
                            projected_logits = self.classifier(transformed_sample)
                        elif self.tabular == True:
                            projected_logits = self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                        projected_pred = torch.round(self.sigmoid(projected_logits))
                        if torch.equal(projected_pred, original_class) == False:
                            break
                    
                    if torch.equal(projected_pred, target_class):
                        return projected_point
                    elif torch.equal(projected_pred, original_class):
                        print('fail. it is still original class. the coef is not proper')
                        return torch.zeros((1, 512)).float().to('cuda')
                
                similar_class = projected_pred
                from_target_class = torch.cat([original_class.to('cuda'), similar_class])
                
                # if original_class == similar_class:
                #     similar_coef = coefs[original_class * self.num_classes + target_class]
                # else:
                for key in self.center_of_target.keys():
                    if torch.equal(key.to('cuda'), similar_class.squeeze()):
                        break
                similar_idx = self.class_direction_finder(matching_class.to('cuda'), from_target_class)
                similar_coef = coefs[similar_idx]

                coef = coef - self.sub_weight * similar_coef
                coef = coef / torch.linalg.norm(coef)

        print('fail. the modifying trials over 100000')
        return torch.zeros((1, 512)).float().to('cuda')
    
    def SVM_maker(self, z, original_class, target_class):
        return self.projection_point_finder(input_z=z, coefs=self.SVM_coefs, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class)

    def finder_maker(self, z, original_class, target_class):
        return self.projection_point_finder(input_z=z, coefs=self.finder_coefs, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class)



class FinderMaker():
    def __init__(self, projector, classifier, generator, args, CTGAN=None, original_image=None, auged_data=None, tabular=False):
        self.finder_coef = 1
        self.latent_dim = args.latent_dim
        self.k_count = args.k_count
        self.num_classes = args.num_classes
        self.projector = projector
        self.classifier = classifier
        self.generator = generator
        self.GMM_component = args.GMM_component
        self.peak_ratio = args.peak_ratio
        self.tabular = tabular
        if CTGAN:
            self.CTGAN = CTGAN
        if args.latent_type == 'training':
            self.auged_data = auged_data
            self.original_image = original_image
            self.logits, self.labels = self.logit_label_finder()
            self.refined_logits_dict, self.refined_latent_dict = self.refined_latent_maker()
        elif args.latent_type == 'random':
            self.refined_logits_dict, self.refined_latent_dict = self.random_refined_latent_maker(latent_iteration=args.latent_iteration, random_latent_std=args.random_latent_std, latent_batch=args.latent_batch)
        self.gmms = self.GMM_maker(refined_latent_dict=self.refined_latent_dict, gmm_component=self.GMM_component)
        self.center_of_target = self.center_of_target_maker(gmms=self.gmms)
        self.SVM_coefs = self.SVM_coef_maker()
        self.finder_coefs = self.finder_coef_maker()
        self.normal_method = args.normal_method
        self.path = args.path
        self.sub_weight = args.sub_weight      
        self.input_latent_path = args.input_latent_path
        self.random_latent_std = args.random_latent_std
        
        
    def refined_latent_maker(self):
        logits_dict = {}
        latent_dict = {}
        with torch.no_grad():
            logits, _ = self.classifier(self.original_image / 255 * 2 - 1)
        for i in range(self.num_classes):
            logits_dict[i] = logits[torch.where(self.labels == i)]
            latent_dict[i] = self.auged_data[torch.where(self.labels == i)]

        refined_logits_dict = {}
        refined_latent_dict = {}
        for i in range(self.num_classes):
            top_idx = logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(logits_dict[i]) * self.peak_ratio)]
            refined_logits_dict[i] = logits_dict[i][top_idx]
            refined_latent_dict[i] = latent_dict[i][top_idx]
        return refined_logits_dict, refined_latent_dict

    def generate_activate_transform(self, input, CTGAN, only_activate=False):
        gen_data = CTGAN._generator(input)
        activated_data = CTGAN._apply_activate(gen_data)
        if only_activate==True:
            return activated_data
        elif only_activate==False:
            inv_transformed_data = CTGAN._transformer.inverse_transform(activated_data.detach().cpu().numpy())
            return inv_transformed_data, activated_data
        

    @torch.no_grad()
    def random_refined_latent_maker(self, latent_iteration, random_latent_std=None, latent_batch=None):
        if random_latent_std is None:
            random_latent_std = self.random_latent_std
        logits_dict = {}
        latent_dict = {}

        if self.tabular==False:
            z_sum = torch.zeros([latent_batch * latent_iteration, self.latent_dim, 1, 1])
        elif self.tabular==True:
            z_sum = torch.zeros([latent_batch * latent_iteration, self.latent_dim])

        logits_from_z = []
        for iter in range(latent_iteration):
            if self.tabular==False:
                z = random_latent_std * torch.randn([latent_batch, self.latent_dim, 1, 1])
            elif self.tabular==True:
                z = random_latent_std * torch.randn([latent_batch, self.latent_dim])
            z_sum[iter * latent_batch : (iter + 1) * latent_batch] = z
            if self.tabular==False:
                logits_from_z.append(self.classifier(self.generator(z.to('cuda')))[0].to('cpu'))
            elif self.tabular==True:
                activated_data = self.generate_activate_transform(z.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
                logits_from_z.append(self.classifier(activated_data).to('cpu'))

        logits_from_z = torch.cat(logits_from_z)
        pred = torch.argmax(logits_from_z, 1)
        for i in range(self.num_classes):
            logits_dict[i] = logits_from_z[torch.where(pred == i)]
            latent_dict[i] = z_sum[torch.where(pred == i)]

        refined_logits_dict = {}
        refined_latent_dict = {}
        for i in range(self.num_classes):
            top_idx = logits_dict[i].max(1).values.sort(descending=True).indices[: int(len(logits_dict[i]) * self.peak_ratio)]
            refined_logits_dict[i] = logits_dict[i][top_idx]
            refined_latent_dict[i] = latent_dict[i][top_idx]
        return refined_logits_dict, refined_latent_dict

    def SVM_coef_maker(self):
        if self.tabular==False:
            SVM_coefs = torch.zeros([self.k_count, 1, self.latent_dim, 1, 1])
        if self.tabular==True:
            SVM_coefs = torch.zeros([self.k_count, 1, self.latent_dim])
        for from_class in range(self.num_classes):
            for to_class in range(self.num_classes):
                svc_model = SVC(C=1.0, kernel='linear')
                sample_origin_target_data = torch.cat([self.refined_latent_dict[from_class], self.refined_latent_dict[to_class]])
                sample_origin_target_label = torch.cat([torch.zeros(len(self.refined_latent_dict[from_class])), torch.ones(len(self.refined_latent_dict[to_class]))])
                sample_origin_target_data = sample_origin_target_data.detach().cpu().numpy()
                sample_origin_target_label = sample_origin_target_label.detach().cpu().numpy()

                svc_model.fit(sample_origin_target_data.squeeze(), sample_origin_target_label)
                temp_coef = svc_model.coef_ / np.linalg.norm(svc_model.coef_)
                temp_coef = temp_coef.astype(float)
                if self.tabular==False:
                    temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim, 1, 1)
                elif self.tabular==True:
                    temp_coef = torch.tensor(temp_coef, device='cuda').reshape(1, self.latent_dim)

                SVM_coefs[from_class * self.num_classes + to_class] = temp_coef
        return SVM_coefs.to('cuda')

    def finder_coef_maker(self):
        if self.tabular==False:
            finder_coefs = torch.zeros([self.k_count, 1, self.latent_dim, 1, 1], device='cuda')
        elif self.tabular==True:
            finder_coefs = torch.zeros([self.k_count, 1, self.latent_dim], device='cuda')
        for idx in range(self.k_count):
            sample = torch.zeros(self.k_count, device='cuda')
            sample[idx] = 1
            if self.tabular==False:
                temp_finder_coef = self.projector(sample).reshape(1, self.latent_dim, 1, 1)
            elif self.tabular==True:
                temp_finder_coef = self.projector(sample).reshape(1, self.latent_dim)
            temp_finder_coef = temp_finder_coef / torch.linalg.norm(temp_finder_coef)

            finder_coefs[idx] = temp_finder_coef
        return finder_coefs.to('cuda')

    def logit_label_finder(self):
        input_image = self.original_image.to('cpu')
        cpu_classifier = self.classifier.to('cpu')
        logits = cpu_classifier(input_image / 255 * 2 - 1)[0]
        softmax_func = torch.nn.Softmax(dim=1)
        softmax_result = softmax_func(logits)
        labels = softmax_result.argmax(1)
        logits = logits.to('cuda')
        labels = labels.to('cuda')
        return logits, labels

    def direction_maker(self, z, original_class, target_class, coef_name):
        if coef_name == 'svm':
            coef = self.SVM_coefs[original_class * self.num_classes + target_class]
        elif coef_name == 'finder':
            coef = self.finder_coefs[original_class * self.num_classes + target_class]
        scalar=0
        for trial in range(1000):
            z_direction, scalar = self.latent_finder(latent=z, 
                                                    coef=coef, 
                                                    target_class=target_class, 
                                                    recent_scalar=scalar, 
                                                    scalar_start=0, 
                                                    scalar_end=10, 
                                                    num=1000, 
                                                    debug_mode=False)

            with torch.no_grad():
                if self.tabular==False:
                    temp_logits, _ = self.classifier(self.generator(z_direction.to('cuda')))
                elif self.tabular==True:
                    temp_activated = self.generate_activate_transform(z_direction.to('cuda'), CTGAN=self.CTGAN, only_activate=True)
                    temp_logits = self.classifier(temp_activated)
                temp_pred = int(torch.argmax(temp_logits))
            if temp_pred == target_class:
                return z_direction
        print('not found in direction')

    @torch.no_grad()
    def latent_finder(self, latent, coef, target_class, recent_scalar, scalar_start=0, scalar_end=100, num=100000, debug_mode=False):
        z = latent
        for scalar in np.linspace(scalar_start, scalar_end, num):
            if scalar == 0:
                continue
            scalar = torch.tensor(scalar, dtype=float).to('cuda')
            scalar_new = scalar + recent_scalar
            # print(f'scalar:{scalar_new}')
            if self.tabular == False:
                temp_latent = (z.to('cuda') + (scalar_new) * coef.reshape(1, self.latent_dim, 1, 1)).float()
                temp_generated_image = self.generator(temp_latent)
                logits, expected_features = self.classifier(temp_generated_image)
            elif self.tabular == True:
                temp_latent = (z.to('cuda') + (scalar_new) * coef.reshape(1, self.latent_dim)).float()
                temp_activated = self.generate_activate_transform(temp_latent, self.CTGAN, only_activate=True)
                logits = self.classifier(temp_activated)
            
            pred = int(torch.argmax(logits, 1))
            if debug_mode == True and self.tabular==False:
                plt.imshow(temp_generated_image.squeeze().detach().cpu())
                plt.title(f'pred_label:{pred}, scalar:{scalar_new:.1} for directing coef to test dataset')
                plt.show()
            if pred == target_class:
                return temp_latent, scalar_new
        return temp_latent, scalar_new

    def GMM_maker(self, refined_latent_dict, gmm_component):
        gmms = {}
        for gmm_class in range(self.num_classes):
            gmms_element = {}
            for element in range(self.latent_dim):
                gmm = GaussianMixture(n_components=gmm_component, random_state=0)
                gmm = gmm.fit(refined_latent_dict[gmm_class].squeeze()[:, element].reshape(-1, 1).detach().cpu().numpy())
                gmms_element[element] = gmm
            gmms[gmm_class] = gmms_element
        return gmms

    def center_of_target_maker(self, gmms):
        target_mode = torch.zeros([self.num_classes, self.latent_dim], device='cuda')
        for from_class in tqdm(range(self.num_classes), desc='center of target maker'):
            for element in range(self.latent_dim):
                sample_element = gmms[from_class][element].sample(10000)
                probabiliy_gmm = gmms[from_class][element].score_samples(sample_element[0])
                max_idx = np.argmax(probabiliy_gmm)
                target_mode[from_class][element] = torch.tensor(sample_element[0][max_idx])
        return target_mode

    @torch.no_grad()
    def pole_point_finder(self, input_z, coefs, refined_latent_dict, original_class, target_class):
        coef = coefs[original_class * self.num_classes + target_class]
        z_point = input_z
        for modifying_trial in range(self.modifying_trial):
            if self.normal_method == 'nearest' :
                idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[target_class] - z_point)**2)).sum(2)))
                nearest_point = refined_latent_dict[target_class][idx_argmin]
                vector_to_cal = (nearest_point - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim, 1, 1).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim, 1, 1)

            elif self.normal_method == 'mean':
                mean_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].mean(0))**2)).sum(2)))
                    mean_dict[i] = refined_latent_dict[i][idx_argmin]
                target_mean = mean_dict[target_class]
                vector_to_cal = (target_mean - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim, 1, 1).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim, 1, 1)
            
            elif self.normal_method == 'mean_point':
                mean_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].mean(0))**2)).sum(2)))
                    mean_dict[i] = refined_latent_dict[i][idx_argmin]
                target_mean = refined_latent_dict[target_class].mean(0)
                vector_to_cal = (target_mean - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim, 1, 1).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim, 1, 1)

            elif self.normal_method == 'median':
                median_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].median(0)[0])**2)).sum(2)))
                    median_dict[i] = refined_latent_dict[i][idx_argmin]
                target_median = median_dict[target_class]
                vector_to_cal = (target_median - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim, 1, 1).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim, 1, 1)
            
            elif self.normal_method == 'median_point':
                median_dict = {}
                for i in range(10):
                    idx_argmin = int(torch.argmin((torch.sqrt((refined_latent_dict[i] - refined_latent_dict[i].median(0)[0])**2)).sum(2)))
                    median_dict[i] = refined_latent_dict[i][idx_argmin]
                target_median = refined_latent_dict[target_class].median(0)[0]
                vector_to_cal = (target_median - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.reshape(1, self.latent_dim, 1, 1).float().squeeze())
                projected_point = z_point + matmul_cal * coef.reshape(1, self.latent_dim, 1, 1)

            elif self.normal_method == 'mode':

                target_mode = self.center_of_target[target_class]
                gmms = self.gmms
                if self.tabular == False:
                    target_mode = target_mode.reshape(1, self.latent_dim, 1, 1).to('cuda')
                elif self.tabular == True:
                    target_mode = target_mode.reshape(1, self.latent_dim).to('cuda')
                vector_to_cal = (target_mode - z_point).float()
                matmul_cal = torch.matmul(vector_to_cal.squeeze(), coef.float().squeeze())
                projected_point = z_point + matmul_cal * coef

            if self.tabular == False:
                projected_logits, _ = self.classifier(self.generator(projected_point.float()))
            elif self.tabular == True:
                projected_logits= self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
            projected_pred = int(torch.argmax(projected_logits))

            if projected_pred != target_class:
                if projected_pred == original_class:
                    for sameclass_trial in range(8000):
                        projected_point = projected_point + coef * self.sub_weight
                        if self.tabular == False:
                            projected_logits, _ = self.classifier(self.generator(projected_point.float()))
                        elif self.tabular == True:
                            projected_logits = self.classifier(self.generate_activate_transform(projected_point.float(), CTGAN=self.CTGAN, only_activate=True))
                        projected_pred = int(torch.argmax(projected_logits))
                        if projected_pred == target_class:
                            break
                    
                    if projected_pred == target_class:
                        return projected_point
                
                similar_class = projected_pred
                if original_class == similar_class:
                    similar_coef = coefs[original_class * self.num_classes + target_class]
                else:
                    similar_coef = coefs[original_class * self.num_classes + similar_class]

                coef = coef + self.sub_weight * (coef - similar_coef)
                coef = coef / torch.linalg.norm(coef)

            if projected_pred == target_class:
                return projected_point

        print('fail')
    
    def SVM_maker(self, z, original_class, target_class):
        return self.pole_point_finder(input_z=z, coefs=self.SVM_coefs, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class)

    def finder_maker(self, z, original_class, target_class):
        return self.pole_point_finder(input_z=z, coefs=self.finder_coefs, refined_latent_dict=self.refined_latent_dict, original_class=original_class, target_class=target_class)


