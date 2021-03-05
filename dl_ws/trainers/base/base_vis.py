import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
cv2.setNumThreads(0)


class BaseVis:
    def __init__(self, config):
        self.c = config

    def visualize_batch_result(self, vis, prefix):
        im_lst = []
        cmap = plt.get_cmap('magma_r')

        for k in sorted(vis.keys()):
            v = vis[k]

            if 'x' in k:
                if 'prenorm' in k:
                    x = np.ascontiguousarray((v[0]).cpu().numpy())
                else:
                    x = np.ascontiguousarray((v[0] * 255).permute(1, 2,
                                                                  0).cpu().numpy())

                cv2.putText(x, k, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                im_lst.append(x.astype(np.uint8))

            elif 'y' in k:
                np_depth = vis[k][0].detach().cpu().squeeze().numpy()
                norm_depth = np_depth  / np.max(np_depth)
                pred_heatmap_orig = np.ascontiguousarray((cmap(norm_depth) *
                                                          255)[:, :, :3])
                pred_heatmap = pred_heatmap_orig.copy()
                cv2.putText(pred_heatmap, k, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                im_lst.append(pred_heatmap.astype(np.uint8))
            else:
                assert False, f'{k} key needs to be dealt with!'

        vis = np.concatenate(im_lst, axis=1)

        if self.writer:
            self.writer.add_image(prefix,
                                  vis.transpose(2, 0, 1), global_step=self.iteration)

            images_dir = os.path.join(self.c.exp_path, 'images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            vis_dir = os.path.join(images_dir, prefix)
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            cv2.imwrite(os.path.join(
                vis_dir, f'{prefix}_{self.iteration}.jpg'), vis)
