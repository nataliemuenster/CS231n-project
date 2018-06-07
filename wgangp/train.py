from tqdm import tqdm
import checkpoints, reporter
from defaults import CHECKPOINTS_DIR, FINALS_DIR, FINAL_IMGS

def train(model, data, dtype, args):
  iter_count = 0
  disc_solver = model.disc_optimizer(args['learning_rate'], args['beta1'], args['beta2'], args['weight_decay'])
  gen_solver = model.gen_optimizer(args['learning_rate'], args['beta1'], args['beta2'], args['weight_decay'])

  starting_epoch = 0
  if args['resume']:
    checkpoints.load_checkpoint(model, CHECKPOINTS_DIR, args['resume'])
    starting_epoch = args['resume'] // len(data)
    iter_count = args['resume']
    print('Loading checkpoint of {name} at {chkpt_iter}'.format(
      name=model.name, chkpt_iter=args['resume']
    ))

  for epoch in tqdm(range(starting_epoch, args['num_epochs']), desc='epochs', position=1):
    for batch_index, batch in tqdm(enumerate(data), desc='iterations', position=2, total=len(data)):
      if args['resume'] and batch_index < iter_count % len(data):
        continue
      try:
        x, _ = batch
      except ValueError:
        x = batch
      
      x = x.type(dtype)  # Batch data
      
      # Calculate number of discriminator iterations
      disc_iterations = args['disc_warmup_iterations'] if (
        batch_index < args['disc_warmup_length'] or batch_index % args['disc_rapid_train_interval'] == 0
      ) else args['disc_iterations']
      
      # Train discriminator
      for _ in range(disc_iterations):
        disc_solver.zero_grad()
        real_data = model.preprocess_data(x).type(dtype)
        noise = model.sample_noise(args['batch_size']).type(dtype)
        disc_loss, fake_images = model.disc_loss(real_data, noise, True)
        disc_loss_gp = disc_loss + model.gradient_penalty(x, fake_images, lambda_val=args['lambda_val'])
        disc_loss_gp.backward()
        disc_solver.step()

      # Train generator
      gen_solver.zero_grad()
      noise = model.sample_noise(args['batch_size']).type(dtype)
      gen_loss = model.gen_loss(noise)
      gen_loss.backward()
      gen_solver.step()

      if iter_count % args['losses_every'] == 0:
        reporter.visualize_scalar(
          -disc_loss.item(),
          'Wasserstein distance between x and g',
          iteration=iter_count,
          env=model.name
        )
        reporter.visualize_scalar(
          gen_loss.item(),
          'Generator loss',
          iteration=iter_count,
          env=model.name
        )

      # send sample images to the visdom server.
      if iter_count % args['images_every'] == 0:
        reporter.visualize_images(
          model.sample_images(args['sample_size']).data,
          'generated samples {}'.format(iter_count // args['images_every']),
          env=model.name
        )

      if iter_count % args['checkpoint_every'] == 0 and iter_count != 0:
        checkpoints.save_checkpoint(model, CHECKPOINTS_DIR, iter_count, args)
      
      iter_count += 1
    args['resume'] = None
  samples = model.sample_images(args['sample_size']).data
  reporter.visualize_images(samples, 'final generated samples', env=model.name)
  checkpoints.save_images(samples, args['tag'])
  checkpoints.save_checkpoint(model, FINALS_DIR, iter_count)

