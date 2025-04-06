# Cross-Pseudo-Supervision-CPS-semi-supervised-segmentation
Cross Pseudo Supervision (CPS) is a semi-supervised learning technique designed to improve semantic segmentation by leveraging unlabeled data more effectively.

Concept

CPS uses two neural networks with the same architecture but different initializations.
Each network generates pseudo labels for the unlabeled data.
These pseudo labels are then used to supervise the other network, creating a cross-supervision mechanism.
Domain Adaptation

By leveraging labeled source domain data, a small amount of labeled target domain data, and a large amount of unlabeled target domain data, CPS bridges the gap between the source and target domains. This improves the model's performance on the target domain.

![image](https://github.com/user-attachments/assets/56ff8953-a95c-421c-9dc4-4d57dbc283d1)


Pseudo Labeling

For the unlabeled target domain data, each network generates pseudo labels based on its predictions. These pseudo labels act as a form of supervision for the other network. The pseudo labels from Network A are used to supervise Network B, and vice versa. This cross-supervision mechanism helps the networks learn from each other's strengths and correct potential biases in their predictions.

def generate_pseudo_labels(model, images):
  with torch.no_grad():
    outputs = model(images)
    psuedo_labels = torch.argmax(outputs, dim=1)
  return psuedo_labels


  Qualitative Evaluation: Visual Inspection

You can visualize the model's predictions to assess segmentation quality by overlaying the predictions on the original images. This is especially useful when ground truth data isn’t available.

![image](https://github.com/user-attachments/assets/ea771fe1-31f2-4d97-8d2d-810299d3bf4a)








