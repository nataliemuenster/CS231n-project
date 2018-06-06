from tqdm import tqdm
import reporter

def train(model, data, dtype, args):
  iter_count = 0
  disc_solver = model.disc_optimizer(args['learning_rate'], args['beta1'], args['beta2'], args['weight_decay'])
  gen_solver = model.gen_optimizer(args['learning_rate'], args['beta1'], args['beta2'], args['weight_decay'])
  for epoch in tqdm(range(args['num_epochs']), desc='epochs', position=1):
    for batch_index, batch in tqdm(enumerate(data, 1), desc='iterations', position=2, total=len(data)):
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
      
      iter_count += 1
  reporter.visualize_images(
    model.sample_images(args['sample_size']).data,
    'final generated samples',
    env=model.name
  )

