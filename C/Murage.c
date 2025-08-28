#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"
//clear && gcc Murage.c -lm -o m.o && ./m.o

typedef struct kibicho_tensor_struct *KibichoTensor;
typedef struct chebyshev_kan_layer_struct *ChebyshevKANLayer;
typedef struct chebyshev_kan_model_struct *ChebyshevKANModel;
struct chebyshev_kan_layer_struct
{
	size_t batchSize;
	size_t inputDimension;
	size_t outputDimension;
	size_t polynomialDegree;
	KibichoTensor inputTensor;
	KibichoTensor polynomialCoefficients;
	KibichoTensor layerWeights;
	KibichoTensor outputTensor;
};

struct chebyshev_kan_model_struct
{
	ChebyshevKANLayer *chebyshevKANLayers;
};

struct kibicho_tensor_struct
{
	size_t size;
	size_t dimensionCount;
	int foundKibichoTensor;
	size_t offsetStart;
	size_t offsetEnd;
	int *shape;
	int *strides;
	float *data;
};

KibichoTensor CreateKibichoTensor()
{
	KibichoTensor tensor = malloc(sizeof(struct kibicho_tensor_struct));
	tensor->dimensionCount = 0;
	tensor->size = 0;
	tensor->dimensionCount = 0;
	tensor->foundKibichoTensor = -1;
	tensor->offsetStart = 0;
	tensor->offsetEnd = 0;
	tensor->shape = NULL;
	tensor->strides = NULL;
	tensor->data = NULL;
	return tensor;
}

void SetInitStrideKibichoTensor(KibichoTensor tensor)
{
	if(tensor)
	{
		tensor->dimensionCount = arrlen(tensor->shape);
		int stride = 1;
		tensor->size = 1;
		arrsetlen(tensor->strides, tensor->dimensionCount);
		for(int i = tensor->dimensionCount - 1; i >= 0; i--)
		{
			tensor->strides[i] = stride;
			stride *= tensor->shape[i];
			tensor->size *= tensor->shape[i];
		}
		
		//Set random values
		for(size_t i = 0; i < tensor->size; i++)
		{
			arrput(tensor->data, ((double)rand() / RAND_MAX - 0.5) * 0.2);
		}
	}
}
void DestroyKibichoTensor(KibichoTensor tensor)
{
	if(tensor)
	{
		if(tensor->data){arrfree(tensor->data);}
		if(tensor->shape){arrfree(tensor->shape);}
		if(tensor->strides){arrfree(tensor->strides);}
		free(tensor);
	}
}

void SetTensorItem_Float(KibichoTensor tensor, int indexLength, size_t *indices, float value)
{
	if(tensor)
	{
		assert(indexLength == tensor->dimensionCount);
		assert(tensor->strides);assert(tensor->data);
		int index = 0;
		for(int i = 0; i < tensor->dimensionCount; i++)
		{
			assert(indices[i] >= 0);
			assert(indices[i] < tensor->shape[i]);
			index += indices[i] * tensor->strides[i];
		}
		assert(index > -1);
		assert(index < tensor->size);
		tensor->data[index] = value;
	}
}

float GetTensorItem_Float(KibichoTensor tensor, int indexLength, size_t *indices)
{
	if(tensor)
	{
		assert(indexLength == tensor->dimensionCount);
		assert(tensor->strides);assert(tensor->data);
		int index = 0;
		for(int i = 0; i < tensor->dimensionCount; i++)
		{
			assert(indices[i] >= 0);
			assert(indices[i] < tensor->shape[i]);
			index += indices[i] * tensor->strides[i];
		}
		assert(index > -1);
		assert(index < tensor->size);
		return tensor->data[index];
	}
}
void PrintKibichoTensor(KibichoTensor tensor)
{
	if(tensor)
	{
		printf("Size: %3ld, Dimensions: %3ld Offsets[%3ld,%3ld] Shape[%3d", tensor->size,tensor->dimensionCount, tensor->offsetStart,tensor->offsetEnd,tensor->shape[0]);
		for(int i = 1; i < tensor->dimensionCount; i++)
		{
			printf(",%3d", tensor->shape[i]);
		}
		printf("] Strides[%3d",tensor->strides[0]);
		for(int i = 1; i < tensor->dimensionCount; i++)
		{
			printf(",%3d", tensor->strides[i]);
		}
		printf("]\n");
	}
}

void PrintMatrix2D(int rows, int cols, float *matrix)
{
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			int index = (i) * (cols) + (j);
			printf("%.3f ", matrix[index]);
		}
		printf("\n");
	}
}

void PrintTensorAsMatrix(KibichoTensor tensor)
{
	assert(tensor->dimensionCount == 2);
	size_t indices[2];
	for(size_t i = 0; i < tensor->shape[0]; i++)
	{
		for(size_t j = 0; j < tensor->shape[1]; j++)
		{
			indices[0] = i;
			indices[1] = j;
			float val = GetTensorItem_Float(tensor, 2, indices);
			printf("%.3f ", val);
		}
		printf("\n");
	}
}


ChebyshevKANLayer CreateChebyshevKANLayer(size_t batchSize,size_t inputDimension,size_t outputDimension, size_t polynomialDegree)
{
	ChebyshevKANLayer layer = malloc(sizeof(struct chebyshev_kan_layer_struct));
	layer->batchSize = batchSize;
	layer->inputDimension = inputDimension;
	layer->outputDimension = outputDimension;
	layer->polynomialDegree = polynomialDegree;
	layer->inputTensor = CreateKibichoTensor();
	layer->polynomialCoefficients = CreateKibichoTensor();
	layer->layerWeights = CreateKibichoTensor();
	layer->outputTensor = CreateKibichoTensor();
	
	arrput(layer->inputTensor->shape, batchSize);
	arrput(layer->inputTensor->shape, inputDimension);
	SetInitStrideKibichoTensor(layer->inputTensor);
	
	arrput(layer->polynomialCoefficients->shape, batchSize);
	arrput(layer->polynomialCoefficients->shape, inputDimension);
	arrput(layer->polynomialCoefficients->shape, polynomialDegree + 1);
	SetInitStrideKibichoTensor(layer->polynomialCoefficients);
	
	arrput(layer->layerWeights->shape, inputDimension);
	arrput(layer->layerWeights->shape, outputDimension);
	arrput(layer->layerWeights->shape, polynomialDegree + 1);
	SetInitStrideKibichoTensor(layer->layerWeights);
	
	arrput(layer->outputTensor->shape, batchSize);
	arrput(layer->outputTensor->shape, outputDimension);
	SetInitStrideKibichoTensor(layer->outputTensor);
	return layer;
	
}

void DestroyChebyshevKANLayer(ChebyshevKANLayer layer)
{
	if(layer)
	{
		DestroyKibichoTensor(layer->outputTensor);
		DestroyKibichoTensor(layer->inputTensor);
		DestroyKibichoTensor(layer->polynomialCoefficients);
		DestroyKibichoTensor(layer->layerWeights);
		free(layer);
	}
}

ChebyshevKANModel CreateChebyshevKANModel()
{
	ChebyshevKANModel model = malloc(sizeof(struct chebyshev_kan_model_struct));
	model->chebyshevKANLayers = NULL;
	return model;
}

void DestroyChebyshevKANModel(ChebyshevKANModel model)
{
	if(model)
	{
		for(size_t i = 0; i < arrlen(model->chebyshevKANLayers); i++)
		{
			DestroyChebyshevKANLayer(model->chebyshevKANLayers[i]);
		}
		arrfree(model->chebyshevKANLayers);
		free(model);
	}
}

double TargetFunction(double x)
{
	return x * (1 - x) * sin((2 * M_PI) / (x + 0.1));
}

double NormalizeInput(double x)
{
	return tanh(x);
}

double ClipGradient(double x, double maxGradient)
{
	if(x > maxGradient) return maxGradient;
	if(x < -maxGradient) return -maxGradient;
	return x;
}



void CompileChebyshevKANModel(ChebyshevKANModel model, size_t batchSize,size_t dimensionLength, size_t *inputDimension, size_t *outputDimension, size_t *polynomialDegree)
{
	if(model && inputDimension && outputDimension && polynomialDegree)
	{
		for(size_t i = 0; i < dimensionLength; i++)
		{
			ChebyshevKANLayer layer = CreateChebyshevKANLayer(batchSize, inputDimension[i], outputDimension[i], polynomialDegree[i]);		
			arrput(model->chebyshevKANLayers, layer);
		}
	}
}

void PrintChebyshevKANModel(ChebyshevKANModel model)
{
	if(model)
	{
		printf("Model Layers : %ld\n", arrlen(model->chebyshevKANLayers));
		for(size_t i = 0; i < arrlen(model->chebyshevKANLayers); i++)
		{
			printf("Layer %ld:\n",i);
			ChebyshevKANLayer layer = model->chebyshevKANLayers[i];
			PrintKibichoTensor(layer->inputTensor);
			PrintKibichoTensor(layer->polynomialCoefficients);
			PrintKibichoTensor(layer->layerWeights);
			PrintKibichoTensor(layer->outputTensor);
			printf("\n");	
		}
	}
}

void ComputeChebyshevLayerOutput(ChebyshevKANLayer layer)
{
    if(layer)
    {
        KibichoTensor T = layer->polynomialCoefficients;  // [b, i, d]
        KibichoTensor C = layer->layerWeights;           // [i, o, d]
        KibichoTensor y = layer->outputTensor; 
        assert(T->dimensionCount == 3);
        assert(C->dimensionCount == 3);
        assert(y->dimensionCount == 2);
        
        size_t B = T->shape[0];  // batch size
        size_t I = T->shape[1];  // input dimension
        size_t D = T->shape[2];  // polynomial degree + 1
        size_t O = C->shape[1];  // output dimension
        
        // Clear output tensor
        for(size_t i = 0; i < y->size; i++) {
            y->data[i] = 0.0f;
        }
        
        // Perform einsum: "bid,iod->bo" = y[b][o] = sum_i sum_d T[b][i][d] * C[i][o][d]
        size_t b_indices[3] = {0};
        size_t c_indices[3] = {0};
        size_t y_indices[2] = {0};
        
        for(size_t b = 0; b < B; b++)
        {
            y_indices[0] = b;
            
            for(size_t o = 0; o < O; o++)
            {
                y_indices[1] = o;
                double sum = 0.0f;
                
                for(size_t i = 0; i < I; i++)
                {
                    b_indices[0] = b;
                    b_indices[1] = i;
                    c_indices[0] = i;
                    c_indices[1] = o;
                    
                    for(size_t d = 0; d < D; d++)
                    {
                        b_indices[2] = d;
                        c_indices[2] = d;
                        
                        float t_val = GetTensorItem_Float(T, 3, b_indices);
                        float c_val = GetTensorItem_Float(C, 3, c_indices);
                        sum += t_val * c_val;
                    }
                }
                
                SetTensorItem_Float(y, 2, y_indices, sum);
            }
        }   
    }
}


double FindChebyshevAtDegree(int degree, double x) 
{
	if(degree == 0) return 1.0;
	if(degree == 1) return x;
	double T0 = 1.0, T1 = x, Tn;
	for (int i = 2; i <= degree; i++)
	{
		Tn = 2.0 * x * T1 - T0;
		T0 = T1;
		T1 = Tn;
	}
	return T1;
}

void FillChebyshevPolynomialTensor(ChebyshevKANLayer layer)
{
	if(layer)
	{
		KibichoTensor T = layer->polynomialCoefficients;  // [b, i, d]
		KibichoTensor C = layer->layerWeights;           // [i, o, d]
		KibichoTensor y = layer->outputTensor; 
		KibichoTensor inputTensor = layer->inputTensor; 
		assert(T->dimensionCount == 3);
		assert(C->dimensionCount == 3);
		assert(y->dimensionCount == 2);
		assert(inputTensor->dimensionCount == 2);
		size_t inputIndexer[2] = {0};
		size_t tIndexer[3] = {0};
		for(size_t i = 0; i < inputTensor->shape[0]; i++)
		{
			for(size_t j = 0; j < inputTensor->shape[1]; j++)
			{
				inputIndexer[0] = i;
				inputIndexer[1] = j;
				
				tIndexer[0] = i;
				tIndexer[1] = j;
				float inputValue = GetTensorItem_Float(inputTensor, 2, inputIndexer);
				for(size_t k = 0; k < layer->polynomialDegree; k++)
				{
					tIndexer[2] = k;
					float polynomial = FindChebyshevAtDegree(k, inputValue);
					SetTensorItem_Float(T, 3, tIndexer, polynomial);
				}
			}
		}
		
	}
}

void TestModel2By2()
{
	srand(543);
	ChebyshevKANModel model = CreateChebyshevKANModel();
	size_t batchSize = 2;
	size_t inputDimension[]   = {2,4,3};
	size_t outputDimension[]  = {1,1,6};
	size_t polynomialDegree[] = {3,7,9};
	size_t dimensionLength0 = sizeof(inputDimension) / sizeof(size_t);
	size_t dimensionLength1 = sizeof(outputDimension) / sizeof(size_t);
	size_t dimensionLength2 = sizeof(polynomialDegree) / sizeof(size_t);
	assert(dimensionLength0 == dimensionLength1);
	assert(dimensionLength1 == dimensionLength2);
	
	//Compile model
	CompileChebyshevKANModel(model, batchSize, dimensionLength0,inputDimension,outputDimension,polynomialDegree);
	PrintChebyshevKANModel(model);
	
	//Forward Pass
	//Set example
	model->chebyshevKANLayers[0]->inputTensor->data[0] = 0.5;
	model->chebyshevKANLayers[0]->inputTensor->data[1] = -1.0;
	model->chebyshevKANLayers[0]->inputTensor->data[2] = 2.0;
	model->chebyshevKANLayers[0]->inputTensor->data[3] = 0.3;
	PrintMatrix2D(2,2,model->chebyshevKANLayers[0]->inputTensor->data);printf("\n");
	
	//Perform Elementwise Input Normalization
	for(int i = 0; i < 4; i++)
	{
		model->chebyshevKANLayers[0]->inputTensor->data[i] = tanh(model->chebyshevKANLayers[0]->inputTensor->data[i]);
	}
	PrintMatrix2D(2,2,model->chebyshevKANLayers[0]->inputTensor->data);printf("\n");
	
	//Calculate Chebyshev polynomials
	FillChebyshevPolynomialTensor(model->chebyshevKANLayers[0]);
	PrintMatrix2D(4,4,model->chebyshevKANLayers[0]->polynomialCoefficients->data);printf("\n");
	
	//Set learnable coefficients
	double temporaryWeight = 0.1;
	for(int i = 0; i < 6; i++)
	{
		model->chebyshevKANLayers[0]->layerWeights->data[i] = temporaryWeight;
		temporaryWeight += 0.1;
	}
	PrintMatrix2D(2,3,model->chebyshevKANLayers[0]->layerWeights->data);printf("\n");
	//Calculate output tensor
	ComputeChebyshevLayerOutput(model->chebyshevKANLayers[0]);
	PrintTensorAsMatrix(model->chebyshevKANLayers[0]->outputTensor);printf("\n");
	DestroyChebyshevKANModel(model);
}

int main()
{
	TestModel2By2();
	return 0;
}
