from time import time


def smart_print(
    start, 
    num_iterations, 
    current_iteration, 
    current_epoch,
    total_epochs,
    gen_loss,
    disc_loss
):
    avg_runtime = (time() - start) / (current_iteration)
    remaining_estimate = avg_runtime * (num_iterations- current_iteration)
    remaining_estimate_min = remaining_estimate / 60
    to_print = f"Epoch {current_epoch}/{total_epochs}. "
    to_print += f"Batch {current_iteration}/{num_iterations}. "
    to_print += f"Eta {remaining_estimate_min: .2f} min. "
    to_print += f"Gen loss: {gen_loss: .4f}, Disc loss: {disc_loss: .4f}.      "

    end = "\r"
    if current_iteration == num_iterations:
        end = "\n"
    
    print(to_print, end=end)