import Augmentor
p = Augmentor.Pipeline('/hampiholidata/Project/Datasets/imdb/Augment/augment_imdb/')
p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=1, min_factor=1.1, max_factor=1.6)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.sample(14630)

