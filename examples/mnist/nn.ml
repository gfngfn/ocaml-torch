open Base
open Torch

(* This should reach ~97% accuracy. *)
let hidden_nodes = 128
let epochs = 1000
let learning_rate = 1e-3

let () =
  let mnist = Mnist_helper.read_files () in
  let { Dataset_helper.train_images; train_labels; _ } = mnist in
  let vs = Var_store.create ~name:"nn" () in
  let linear1 =
    Layer.linear vs hidden_nodes ~activation:Relu ~input_dim:Mnist_helper.image_dim
  in
  let linear2 = Layer.linear vs Mnist_helper.label_count ~input_dim:hidden_nodes in
  let adam = Optimizer.adam vs ~learning_rate in
  let model xs = Layer.forward linear1 xs |> Layer.forward linear2 in
  Stdio.print_endline "!!!! NN A";
  for index = 1 to epochs do
    (* Compute the cross-entropy loss. *)
    let loss =
      Tensor.cross_entropy_for_logits (model train_images) ~targets:train_labels
    in
    Stdio.print_endline ("!!!! NN B " ^ Int.to_string index);
    Optimizer.backward_step adam ~loss;
    Stdio.print_endline ("!!!! NN C " ^ Int.to_string index);
    if index % 50 = 0
    then (
      Stdio.print_endline ("!!!! NN D " ^ Int.to_string index);
      (* Compute the validation error. *)
      let test_accuracy =
        Dataset_helper.batch_accuracy mnist `test ~batch_size:1000 ~predict:model
      in
      Stdio.print_endline ("!!!! NN E " ^ Int.to_string index);
      Stdio.printf
        "%d %f %.2f%%\n%!"
        index
        (Tensor.float_value loss)
        (100. *. test_accuracy));
    Stdlib.Gc.full_major ()
  done
;;
