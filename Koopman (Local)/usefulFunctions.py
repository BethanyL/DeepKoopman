def pytorch_training(model, x_data, y_data, nb_epochs, learning_rate, criterion, optimizer):
  for epoch in range(0, nb_epochs):
    ### TODO: implemente the training strategy using pytorch functions
    # Forward pass
    y_pred = model(x_data)
    # Compute Loss
    loss = criterion(y_pred, y_data)
    # Zero gradient
    optimizer.zero_grad()
    # Back-propagation
    loss.backward()
    # One-step gradient
    optimizer.step()
    ###
    if epoch % 1000 == 0:
      print(f"Epoch {epoch} ===== loss {loss}")
  return y_pred, model
  