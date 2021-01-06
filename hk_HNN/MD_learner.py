
class MD_learner:

    def __init__(self,linear_integrator):

        self.linear_integrator = linear_integrator
        self.nepoch = nepoch
        self.optimizer = ...


    # phase_space consist of minibatch data
    # pb is boundary condition
    def train(self,**state,phase_space_label,pb):

        pairwise_hnn = state['pairwise_hnn']
        pairwise_hnn.train()

        for e in range(self.nepoch):

            q_list_predict, p_list_predict = self.linear_integrator.integrate(**state)
            loss = self.loss(phase_space_label,pb,q_list_predict,p_list_predict)

            self._optimizer.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
            loss.backward()  # backward pass : compute gradient of the loss wrt model parameters
            train_loss = loss.item()  # get the scalar output
            self._optimizer.step()

    def step(self,phase_space,pb,tau):
        pairwise_hnn.eval()
        q_list_predict, p_list_predict = self.linear_integrator.integrate(**state)
        return q_list_predict,p_list_predict

    def loss(self,...):