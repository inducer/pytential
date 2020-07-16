Merge "betterplane.brep";

Mesh.CharacteristicLengthMax = 0.;
Mesh.ElementOrder = 2;
Mesh.CharacteristicLengthExtendFromBoundary = 0;

// 2D mesh optimization
// Mesh.Lloyd = 1;

l_superfine() = Unique(Abs(Boundary{ Surface{
    27, 25, 17, 13, 18  }; }));
l_fine() = Unique(Abs(Boundary{ Surface{ 2, 6, 7}; }));
l_coarse() = Unique(Abs(Boundary{ Surface{ 14, 16  }; }));

// p() = Unique(Abs(Boundary{ Line{l_fine()}; }));
// Characteristic Length{p()} = 0.05;

Field[1] = Distance;
Field[1].NNodesByEdge = 100;
Field[1].EdgesList = {l_superfine()};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 0.075;
Field[2].LcMax = 0.2;
Field[2].DistMin = 0.1;
Field[2].DistMax = 0.4;

Field[3] = Distance;
Field[3].NNodesByEdge = 100;
Field[3].EdgesList = {l_fine()};

Field[4] = Threshold;
Field[4].IField = 3;
Field[4].LcMin = 0.1;
Field[4].LcMax = 0.2;
Field[4].DistMin = 0.15;
Field[4].DistMax = 0.4;

Field[5] = Distance;
Field[5].NNodesByEdge = 100;
Field[5].EdgesList = {l_coarse()};

Field[6] = Threshold;
Field[6].IField = 5;
Field[6].LcMin = 0.15;
Field[6].LcMax = 0.2;
Field[6].DistMin = 0.2;
Field[6].DistMax = 0.4;

Field[7] = Min;
Field[7].FieldsList = {2, 4, 6};

Background Field = 7;
