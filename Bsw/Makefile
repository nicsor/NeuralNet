INC = -Iinclude
SRC_DIR := src
OBJ_DIR := obj
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
LDFLAGS := 
CPPFLAGS := 
CXXFLAGS := 

retea: $(OBJ_FILES)
	g++ $(LDFLAGS) $(INC) -std=c++17 -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(OBJ_DIR)
	g++ $(CPPFLAGS) $(INC) $(CXXFLAGS) -std=c++17 -c -o $@ $<
	
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) retea
