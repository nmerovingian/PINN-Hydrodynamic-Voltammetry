
import math
def F(theta):
    return math.sqrt(3.0)/(4.0*math.pi)*math.log((1.0+theta**(1.0/3.0))**3.0/(1.0+theta)) + 3.0/(2.0*math.pi)*math.atan((2.0*theta**(1.0/3.0)-1.0)/(math.sqrt(3.0))) + 0.25

def N_MB(theta,lambda_):
    N = 1.0+lambda_**(2.0/3.0)*(1.0-F(theta)) - (1.0+theta+lambda_)**(2.0/3.0)*(1.0-F((theta/lambda_)*(1.0+theta+lambda_))) - F(theta/lambda_)
    return N

def N_Cook(theta,lambda_):
    N = 1.0 - (1.0+theta+lambda_)*F((1.0+theta)/lambda_) + (theta+lambda_)*F(theta/lambda_) + (3.0**(1.5)*lambda_**(2.0/3.0))/(2.0*math.pi)*((1.0+theta)**(1.0/3.0)-lambda_**(1.0/3.0))
    return N


print(N_MB(0.1,1.0))



