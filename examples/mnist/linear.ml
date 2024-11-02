(* Linear model for the MNIST dataset.
   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data' directory.
   train-images-idx3-ubyte.gz
   train-labels-idx1-ubyte.gz
   t10k-images-idx3-ubyte.gz
   t10k-labels-idx1-ubyte.gz

   This should reach ~92% accuracy on the test dataset.
*)
open Base
open Torch

let learning_rate = Tensor.f 1.

let display_shape title tensor =
  Stdio.print_endline ("SHAPE (" ^ title ^ "):");
  List.iter
    ~f:(fun n -> Stdio.print_endline ("- " ^ Int.to_string n))
    (Tensor.shape tensor)

let () =
  let { Dataset_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let ws = Tensor.zeros Mnist_helper.[ image_dim; label_count ] ~requires_grad:true in
  let bs = Tensor.zeros [ Mnist_helper.label_count ] ~requires_grad:true in
  let model xs = Tensor.(mm xs ws + bs) in
  display_shape "ws" ws;
  display_shape "bs" bs;
  for index = 1 to 10 do
    (* Compute the cross-entropy loss. *)
    let loss =
      Tensor.cross_entropy_for_logits (model train_images) ~targets:train_labels
    in
    Tensor.backward loss;
    (* Apply gradient descent, [no_grad f] runs [f] with gradient tracking disabled. *)
    Tensor.(
      no_grad (fun () ->
        ws -= (grad ws * learning_rate);
        bs -= (grad bs * learning_rate)));
    Tensor.zero_grad ws;
    Tensor.zero_grad bs;
    (* Compute the validation error. *)
    let got = model test_images in
    let estimated = Tensor.argmax got in
    display_shape "test_images" test_images;
    display_shape "got" got;
    display_shape "estimated" estimated;
    display_shape "test_labels" test_labels;
    let test_accuracy =
      Tensor.(estimated = test_labels)
      |> Tensor.to_kind ~kind:(T Float)
      |> Tensor.sum
      |> Tensor.float_value
      |> fun sum -> sum /. Float.of_int (Tensor.shape test_images |> List.hd_exn)
    in
    Stdio.printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. test_accuracy);
    Stdlib.Gc.full_major ()
  done
;;
