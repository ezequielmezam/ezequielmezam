var cubeRotation = 0.0;
var modelMatrix      = mat4.create();
var viewMatrix       = mat4.create();
var projectionMatrix = mat4.create();
const fieldOfView = 45 * Math.PI / 180;
const zNear = 0.1;
const zFar = 1000.0;

var eye    = vec3.fromValues(0.0, 0.0, 20.0);
var center = vec3.fromValues(0, 0, 0);
var up     = vec3.fromValues(0, 1, 0);
var luzpos = vec3.fromValues(0.0, 700.0, 0.0);

var cant_puntos_esfera;

let canvas = document.createElement('canvas');
canvas.width =window.innerWidth;
canvas.height=window.innerHeight;
document.body.appendChild(canvas);

var needCapture = false;

//const canvas = document.querySelector('#glcanvas');
const gl = canvas.getContext('webgl2');//, {preserveDrawingBuffer: true} para hacer captura


var cuerpo1 = new Array();
var cuerpo2 = new Array();
var cuerpo3 = new Array();

//var u = [ 0.97000436, -0.24308753, 0.0, -0.466203685 , -0.43236573, 0.0, -0.97000436,  0.24308753, 0.0, -0.466203685 , -0.43236573, 0.0, 0.0, 0.0, 0.0,  0.93240737  ,  0.86473146, 0.0  ];

var u = [ 0.97000436, -0.24308753, 0.0, -0.466203685 , -0.43236573, 0.0, -0.97000436,  0.24308753, 0.0, -0.466203685 , -0.43236573, 0.0, 0.0, 0.0, 0.0,  0.93240737  ,  0.86473146, 0.0  ];
var du= [  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  ];

const h = 0.01;
const a = [ h/2.0, h/2.0, h,   0.0   ];
const b = [ h/6.0, h/3.0, h/3.0, h/6.0 ];

let textura;
var cuboVAO;

main();


function main() {


  if (!gl) {
    alert('Unable to initialize WebGL.');
    return;
  }

  const vsSource = `#version 300 es

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec3 ColorVertice;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float u_time;


vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0; }

float mod289(float x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0; }

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+1.0)*x);
}

float permute(float x) {
     return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float taylorInvSqrt(float r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 grad4(float j, vec4 ip)
  {
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
  vec4 p,s;

  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
  s = vec4(lessThan(p, vec4(0.0)));
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;

  return p;
  }

// (sqrt(5) - 1)/4 = F4, used once below
#define F4 0.309016994374947451

float snoise(vec4 v)
  {
  const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
                        0.276393202250021,  // 2 * G4
                        0.414589803375032,  // 3 * G4
                       -0.447213595499958); // -1 + 4 * G4

// First corner
  vec4 i  = floor(v + dot(v, vec4(F4)) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;
  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
//  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;
//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;
  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C.xxxx
  //  x1 = x0 - i1  + 1.0 * C.xxxx
  //  x2 = x0 - i2  + 2.0 * C.xxxx
  //  x3 = x0 - i3  + 3.0 * C.xxxx
  //  x4 = x0 - 1.0 + 4.0 * C.xxxx
  vec4 x1 = x0 - i1 + C.xxxx;
  vec4 x2 = x0 - i2 + C.yyyy;
  vec4 x3 = x0 - i3 + C.zzzz;
  vec4 x4 = x0 + C.wwww;

// Permutations
  i = mod289(i);
  float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute( permute( permute( permute (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

// Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));

// Mix contributions from the five corners
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  m0 = m0 * m0;
  m1 = m1 * m1;
  return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
               + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;

  }



void main(){
	
    vec3 pos_nueva;
    vec3 normal_nueva;	
    vec2 pos_horizontal;

    //pos_nueva = (1.0+0.5*sin(u_time))*aPos;
    //pos_nueva = aPos - (1.0+sin(u_time))*aNormal;
    //pos_nueva = aPos - 0.25*(1.0 + sin(u_time))*aNormal;

    //pos_nueva = aPos + sin(u_time)*vec3(0.0, 1.0, 0.0);


    //pos_horizontal = vec2(aPos.x, aPos.z);

    //pos_nueva = aPos;
    //pos_nueva.y = snoise(sin(u_time)*0.5*pos_horizontal);

    //pos_nueva = aPos + snoise*vec3(0.0, 1.0, 0.0);

    //3D PERLIN NOISE
    //pos_nueva = vec3(  aPos.x, 0.5*(1.0 + sin(u_time)), aPos.z  );
    //pos_nueva = vec3(  u_time, u_time, u_time  );// aPos.x, aPos.y, aPos.z
    //pos_nueva.y = cnoise( pos_nueva );
	//pos_nueva = aPos+ 0.2*cnoise( 4.0*aPos )*aNormal + 0.1*cnoise( 8.0*aPos )*aNormal+ 0.05*cnoise( 16.0*aPos )*aNormal;
	
	float tiempo = 0.25*u_time;
	pos_nueva  = aPos + 0.2*snoise( vec4(aPos, tiempo ) )*aNormal;
	pos_nueva +=      0.1*snoise( 2.0*vec4(aPos, tiempo ) )*aNormal;
	pos_nueva +=     0.05*snoise( 4.0*vec4(aPos, tiempo ) )*aNormal;
	pos_nueva +=    0.025*snoise( 8.0*vec4(aPos, tiempo ) )*aNormal;
	pos_nueva +=  0.0125*snoise( 16.0*vec4(aPos, tiempo ) )*aNormal;
	
	//pos_nueva = pos_nueva + 0.5*cnoise( vec3( length(vec2(pos_nueva.x, pos_nueva.z)),pos_nueva.y, u_time) )*aNormal;
	
	//pos_nueva = pos_nueva + 0.5*cnoise( vec3( pos_nueva.x, pos_nueva.y, u_time) )*aNormal;
	//pos_nueva = pos_nueva + 0.2*cnoise( 4.0*pos_nueva )*aNormal + 0.1*cnoise( 8.0*pos_nueva )*aNormal+ 0.05*cnoise( 16.0*pos_nueva )*aNormal;
	


	
	vec3 delta1, delta2;
	float r = length(aPos);

	float theta = acos(aPos.y / r);
	float phi = atan(aPos.z, aPos.x);
	delta1.x = r * sin(theta+0.05) * cos(phi);
	delta1.y = r * cos(theta+0.05);
	delta1.z = r * sin(theta+0.05) * sin(phi);
	delta2.x = r * sin(theta) * cos(phi+0.1);
	delta2.y = r * cos(theta);
	delta2.z = r * sin(theta) * sin(phi+0.1);
	
	delta1 = delta1 + 0.2*snoise( vec4(delta1, u_time ) )*normalize(delta1);
	delta2 = delta2 + 0.2*snoise( vec4(delta2, u_time ) )*normalize(delta2);
	
	
	
	normal_nueva = normalize(cross(delta1-aPos,delta2-aPos));

	
	//Normal = vec3(normal_nueva.x, normal_nueva.y, normal_nueva.z);
	





	FragPos = vec3(model * vec4(pos_nueva, 1.0));				//aPos
	Normal = mat3(transpose(inverse(model))) * aNormal ;	//normal_nueva|aNormal
	 

	ColorVertice = aColor;

	TexCoord     = vec2(aTexCoord.x, aTexCoord.y);

	gl_Position  = projection * view * vec4(FragPos, 1.0);
}
  `;

  const fsSource = `#version 300 es
precision highp float;

in vec3 FragPos;
in vec3 Normal;
in vec3 ColorVertice;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture1;
uniform sampler2D texture2;

uniform float AlfaUL;
uniform int UsarAtributoColor;


struct Material {
	vec3  ambient;
	vec3  diffuse;
	vec3  specular;
	float shininess;
};


struct Light {
	vec3 position;

	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	float constant;
	float linear;
	float quadratics;
};


uniform Material material;
uniform Light luz;
uniform vec3 viewPos;
uniform float u_time;

uniform vec3 cuerPos;
uniform mat4 model;


vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 palette( in float t)
			{
					vec3 a = vec3(0.5, 0.5, 0.5);
					vec3 b = vec3(0.5, 0.5, 0.5);
					vec3 c = vec3(1.0, 1.0, 1.0);
					vec3 d = vec3(0.0, 0.33, 0.67);
					return a + b*cos( 6.28318*(c*t+d) );
			}

void main(){
	
	// Calculate the partial derivatives of the fragment position
    vec2 ddx = dFdx(FragPos.xy);
    vec2 ddy = dFdy(FragPos.xy);

    // Calculate the normal using the partial derivatives
    vec3 ddx_fragPos3 = vec3(ddx, dFdx(FragPos.z));
    vec3 ddy_fragPos3 = vec3(ddy, dFdy(FragPos.z));
    vec3 reconstructedNormal = normalize(cross(ddx_fragPos3, ddy_fragPos3));

	vec3 varColorVertice = vec3(0.0);
	
	vec3 ambient, diffuse, specular;


    // ambient
    //vec3 ambient = luz.ambient * material.ambient;


    // diffuse
    vec3 norm = normalize(reconstructedNormal);//Normal|reconstructedNormal
    vec3 luzDir = normalize(luz.position - FragPos);
    float diff = max(dot(norm, luzDir), 0.0);
    //vec3 diffuse = luz.diffuse* ( diff * material.diffuse);


    // specular
    vec3 vistaDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-luzDir, norm);
    vec3 halfwayDir = normalize(luzDir + vistaDir);

    //float spec = pow(max(dot(vistaDir, reflectDir), 0.0), material.shininess);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), material.shininess);

    specular = luz.specular * (spec * material.specular );



	//attenuation
	//float distance = length(luz.position - FragPos);
	//float attenuation = 1.0f/(luz.constant + luz.linear*distance + luz.quadratics*distance*distance );



	if(UsarAtributoColor == 1){

        //varColorVertice = 0.5+ 0.5*(1.0+sin(10.0*u_time))*ColorVertice;//video publicado
        //varColorVertice = vec3(0.5*(1.0+sin(u_time)), 1.0, 1.0);

        //varColorVertice = vec3(fract(0.25*u_time), 0.5 + 0.1*sin(u_time), 1.0);
        //varColorVertice = vec3(fract(0.25*FragPos.y+0.25*u_time), 0.5,  1.0 );
        //varColorVertice = vec3(fract(0.05*length(FragPos)+ u_time), 0.5 , 1.0 );       //fract(sqrt(FragPos.x*FragPos.x+FragPos.z*FragPos.z) );
		 //varColorVertice = vec3(0.5*sin(u_time)+0.5, 0.5, 1.0);
		 //varColorVertice = hsv2rgb( varColorVertice );
		 //varColorVertice = vec3(0.5*sin(u_time)+0.5, 0.5, 0.0);
	
		
		//varColorVertice = vec3(length(FragPos-vec3(model * vec4(0.0,0.0,0.0, 1.0))), 0.5, 0.5);//sqrt(FragPos.x*FragPos.x+FragPos.y*FragPos.y+FragPos.z*FragPos.z)
		//varColorVertice = hsv2rgb(varColorVertice);
		
		//varColorVertice = vec3(ColorVertice);
		varColorVertice = palette(  sin(u_time) + dot(Normal, FragPos-vec3(model * vec4(0.0,0.0,0.0, 1.0))) );//sin(u_time) +


        ambient  = luz.ambient * varColorVertice;
	diffuse  = luz.diffuse * diff * varColorVertice;

        ambient  = luz.ambient * texture(texture1, TexCoord).rgb;
        diffuse  = luz.diffuse *  diff * texture(texture1, TexCoord).rgb ;





		}
	else{

        ambient = luz.ambient * material.ambient;
		diffuse = luz.diffuse* ( diff * material.diffuse);
		


	};


    vec3 result = ambient + diffuse + specular;

	FragColor = vec4(result, AlfaUL);

		}
	  `;

	  const shaderProgram = initShaderProgram(vsSource, fsSource);


	  const programInfo = {
		program: shaderProgram,
		attribLocations: {
		  vertexPosition: gl.getAttribLocation(shaderProgram, 'aPos'),
		  vertexColor: gl.getAttribLocation(shaderProgram, 'aColor'),
		  vertexNormal: gl.getAttribLocation(shaderProgram, 'aNormal'),
		  vertexTexture: gl.getAttribLocation(shaderProgram, 'aTexCoord'),	  
		},
		uniformLocations: {      
		  modelMatrix: gl.getUniformLocation(shaderProgram, 'model'),
		  viewMatrix: gl.getUniformLocation(shaderProgram, 'view'),
		  projectionMatrix: gl.getUniformLocation(shaderProgram, 'projection'),
		  uniformTime: gl.getUniformLocation(shaderProgram, 'u_time'),
		  uniformPosition: gl.getUniformLocation(shaderProgram, 'viewPos'),
		  uniformLightPos: gl.getUniformLocation(shaderProgram, 'luz.position'),
		  uniformLightAmbient: gl.getUniformLocation(shaderProgram, 'luz.ambient'),
		  uniformLightDiff: gl.getUniformLocation(shaderProgram, 'luz.diffuse'),
		  uniformLightSpec: gl.getUniformLocation(shaderProgram, 'luz.specular'),
		  uniformLightConst: gl.getUniformLocation(shaderProgram, 'luz.constant'),
		  uniformLightLinear: gl.getUniformLocation(shaderProgram, 'luz.linear'),
		  uniformLightQuad: gl.getUniformLocation(shaderProgram, 'luz.quadratics'),	  
		  uniformMaterialtAmbient: gl.getUniformLocation(shaderProgram, 'material.ambient'),
		  uniformMaterialDiff: gl.getUniformLocation(shaderProgram, 'material.diffuse'),
		  uniformMaterialSpec: gl.getUniformLocation(shaderProgram, 'material.specular'),
		  uniformMaterialShini: gl.getUniformLocation(shaderProgram, 'material.shininess'),
		  uniformUsarAtributoColor: gl.getUniformLocation(shaderProgram, 'UsarAtributoColor'),
		  uniformAlphaChannel: gl.getUniformLocation(shaderProgram, 'AlfaUL'),
		  uniformCuerPosition: gl.getUniformLocation(shaderProgram, 'cuerPos'),
		  uniformImageLocation1: gl.getUniformLocation(shaderProgram,'texture1'),
		  uniformImageLocation2: gl.getUniformLocation(shaderProgram,'texture2')
		  
		  
		}
	  };
  

  
	  const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
	  
	  mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);
	  mat4.lookAt(viewMatrix, eye, center, up);


	  initBuffers(gl, programInfo);
	  const vao = CrearEsferaVAO(gl, programInfo);

	  cuboVAO = CrearCuboVAO(gl, programInfo);

	  // Load texture
	  textura = loadTexture(gl, "tierra.jpg");
	  // Flip image pixels into the bottom-to-top order that WebGL expects.
	  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

	  var then = 0;

	  function render(now)
		{
		now *= 0.001;  // convertir a segundos
		const deltaTime = now - then;
		then = now;

		drawScene(gl, programInfo, vao, deltaTime, now);

		requestAnimationFrame(render);
		}
	  
	  requestAnimationFrame(render);
}//fin de la declaración de la función main()

function CrearCuboVAO(gl, programInfo){
	
	
	const positions = [
    // Front face
    -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,

    // Back face
    -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,

    // Top face
    -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,

    // Bottom face
    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,

    // Right face
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,

    // Left face
    -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
	];
	
	const _normales = [
	// Front face:
	0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
	// Back face:
	0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
	// Top face:
	0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
	// Bottom face:
	0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
	// Right face:
	1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
	// Left face:
	-1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0
	];
  
  
  const faceColors = [
    [1.0, 1.0, 1.0, 1.0], // Front face: white
    [1.0, 0.0, 0.0, 1.0], // Back face: red
    [0.0, 1.0, 0.0, 1.0], // Top face: green
    [0.0, 0.0, 1.0, 1.0], // Bottom face: blue
    [1.0, 1.0, 0.0, 1.0], // Right face: yellow
    [1.0, 0.0, 1.0, 1.0], // Left face: purple
  ];

  // Convert the array of colors into a table for all the vertices.

  var colors = [];

  for (var j = 0; j < faceColors.length; ++j) {
    const c = faceColors[j];
    // Repeat each color four times for the four vertices of the face
    colors = colors.concat(c, c, c, c);
  }
  
  
    const textureCoordinates = [
    // Front
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
    // Back
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
    // Top
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
    // Bottom
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
    // Right
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
    // Left
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
  ];
  
  
  const indices = [
    0,
    1,
    2,
    0,
    2,
    3, // front
    4,
    5,
    6,
    4,
    6,
    7, // back
    8,
    9,
    10,
    8,
    10,
    11, // top
    12,
    13,
    14,
    12,
    14,
    15, // bottom
    16,
    17,
    18,
    16,
    18,
    19, // right
    20,
    21,
    22,
    20,
    22,
    23, // left
  ];	
	const _vao = CrearVAO(gl, programInfo, positions, _normales, colors, textureCoordinates, indices);
	return _vao;
	
}
function CrearEsferaVAO(gl, programInfo){

	//El límite máximo para el valor de un índice es 2^16 = 65k, por lo tanto no pueden hacerse una drawcall con una lista de índices que supere este valores,
    //Por tanto la cantidad de puntos diferentes que usa la maya no puede ser mayor a 65k y si es una maya cuadrada serían 256x256 como máximo o 362x181 para mantener un lado el doble que el alto

    var num_lon = 362, num_lat = 181, num_puntos;
	var lat, lon, x, y, z;

    num_puntos = num_lat*num_lon + 2;//sumar 2 por los polos
    cant_puntos_esfera = 2*3*num_lon*num_lat;


    var vertices = new Array();	//	3*num_puntos
    var normales = new Array();	//	3*num_puntos
    var  colores = new Array();	//	3*num_puntos
    var texturas = new Array();	//	3*num_puntos
    var  indices = new Array();	//	2*3*(num_lon)*(num_lat) //en webgl existe un límite de 65k para los valores de los indices no así para la cantidad de indices 



    vertices.push(0.0, 0.0, 1.0);
    normales.push(0.0, 0.0, 1.0);
	colores.push(Math.random(), Math.random(), Math.random());

    for ( let _lat = 1; _lat <= num_lat; _lat++)
        for ( let _lon = 0; _lon < num_lon; _lon++){

            //lat = 0.5*Math.PI - (Math.PI*paral)/(num_lat+1);
            lon = (2.0*Math.PI*_lon)/num_lon;

	    lat = Math.PI - Math.PI*_lat/(num_lat+1);


            //x = Math.cos(lon)*Math.cos(lat);
            //y = Math.sin(lon)*Math.cos(lat);
            //z = Math.sin(lat);		
            //vertices.push(x, y, z);
	    //normales.push(x, y, z);

	    x = Math.sin(lat)*Math.cos(lon);
            y = Math.sin(lat)*Math.sin(lon);
            z = Math.cos(lat);

            vertices.push(-x, z, y);
	    normales.push(-x, z, y);

		
	var rgb = toRgb(360*(0.5*x+0.5), 0.5, 0.5);
	colores.push(rgb.red, rgb.green, rgb.blue );			
	//colores.push(Math.random(), Math.random(), Math.random());
    }
    vertices.push(0.0, 0.0, -1.0);
	normales.push(0.0, 0.0, -1.0);
	colores.push(Math.random(), Math.random(), Math.random());

	texturas.push( 0.5, 0.0);
	for( let j = 0 ; j < num_lat ; j++)
	        for( let i = 0 ; i < num_lon ; i++){
            		texturas.push( i/num_lon, (j+1)/num_lat );
           		}
	texturas.push( 0.5, 1.0);


    //INDICES:
    var v1, v2, v3, v4;
    //Polo norte:
    for( let i = 1 ; i < num_lon; i++){

            v1 = 0;
            v2 = i;
            v3 = v2 + 1;
            indices.push(v1);
            indices.push(v2);
            indices.push(v3);

        }
    indices.push(0);
    indices.push(v3);
    indices.push(1);



    //BANDA CENTRAL
    for ( let i = 0; i < num_lat - 1; i++)
        {
        for ( let j = 1; j < num_lon; j++){

            v1 = j + i*num_lon;
            v2 = v1 + num_lon;
            v3 = v1 + 1;
            indices.push(v1);
            indices.push(v2);
            indices.push(v3);

            v4 = v2 + 1;
            indices.push(v3);
            indices.push(v2);
            indices.push(v4);

            }
        v1 = v3;
        v2 = v4;
        v3 = v1 - num_lon + 1;
        indices.push(v1);
        indices.push(v2);
        indices.push(v3);
        v4 = v2 - num_lon + 1;
        indices.push(v3);
        indices.push(v2);
        indices.push(v4);


        }
    //Polo sur:

    for( let i = 0 ; i < num_lon -1; i++){

            v1 = num_puntos - 1 - num_lon  + i;
            v2 = v1 + 1;
            v3 = num_puntos - 1;
            indices.push(v1);
            indices.push(v2);
            indices.push(v3);
        }
    indices.push(v2);
    indices.push(v2 - num_lon + 1);
    indices.push(num_puntos - 1);
	
	
    var _vertices = new Float32Array(3*num_puntos);	//	3*num_puntos
    var _normales = new Float32Array(3*num_puntos);	//	3*num_puntos
    var  _colores = new Float32Array(3*num_puntos);	//	3*num_puntos
    var _texturas = new Float32Array(2*num_puntos);
    var  _indices = new Uint16Array(2*3*(num_lon)*(num_lat));	//	2*3*(num_lon)*(num_lat)
	
	for(let i = 0 ; i < vertices.length; i++){
		_vertices[i] = vertices[i];
		_normales[i] = normales[i];
		_colores[i]  = colores[i];
		
		}
	for(let i = 0 ; i < texturas.length; i++){
	    _texturas[i] = texturas[i];		
		}
	for(let i = 0 ; i < indices.length; i++){
		_indices[i] = indices[i];
		
		}
	
	const vao = CrearVAO(gl, programInfo, _vertices, _normales, _colores, _texturas, _indices);
	return vao;
};

function CrearVAO(gl, programInfo, vertices, normales, colores, texturas, indices ){
	
  var vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  var numComponents = 3;
  var type = gl.FLOAT;
  var normalize = false;
  var stride = 0;
  var offset = 0;

  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  gl.enableVertexAttribArray( programInfo.attribLocations.vertexPosition);
  gl.vertexAttribPointer( programInfo.attribLocations.vertexPosition, numComponents, type, normalize, stride, offset);
  

  const colorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colores), gl.STATIC_DRAW);
  gl.enableVertexAttribArray( programInfo.attribLocations.vertexColor);
  gl.vertexAttribPointer( programInfo.attribLocations.vertexColor, numComponents, type, normalize, stride, offset);
  
  
  const normalBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normales), gl.STATIC_DRAW);
  gl.enableVertexAttribArray( programInfo.attribLocations.vertexNormal);
  gl.vertexAttribPointer( programInfo.attribLocations.vertexNormal, numComponents, type, normalize, stride, offset);
  
  const textureBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, textureBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texturas), gl.STATIC_DRAW);
  gl.enableVertexAttribArray( programInfo.attribLocations.vertexTexture);
  gl.vertexAttribPointer( programInfo.attribLocations.vertexTexture, 2, type, normalize, stride, offset);

  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
	  

  gl.bindVertexArray(null);
  
  return vao;	
	
};



function initBuffers(gl, programInfo) {
	
  console.log( gl.getParameter(gl.MAX_ELEMENT_INDEX) + " 1 " );
  console.log( gl.getParameter(gl.MAX_ELEMENTS_VERTICES) + " 2" );
  console.log( gl.getParameter(gl.MAX_ELEMENTS_INDICES) + " 3" );

  gl.useProgram(programInfo.program);
  
  let ambienColor   = vec3.fromValues( 1.0 , 1.0 , 1.0 );
  let diffuseColor  = vec3.fromValues( 0.5 , 0.5 , 0.5 );
  let specularColor = vec3.fromValues( 1.0 , 1.0 , 1.0 );
  let shininess     = 64.0;
  
  gl.uniform3fv( programInfo.uniformLocations.uniformLightAmbient, ambienColor );
  gl.uniform3fv( programInfo.uniformLocations.uniformLightDiff, diffuseColor );
  gl.uniform3fv( programInfo.uniformLocations.uniformLightSpec, specularColor );
  gl.uniform1f( programInfo.uniformLocations.uniformLightConst, 1.0 );
  gl.uniform1f( programInfo.uniformLocations.uniformLightLinear, 0.0014 );
  gl.uniform1f( programInfo.uniformLocations.uniformLightQuad, 0.000007 );  
  
  gl.uniform3fv( programInfo.uniformLocations.uniformMaterialtAmbient, ambienColor );
  gl.uniform3fv( programInfo.uniformLocations.uniformMaterialDiff, diffuseColor );
  gl.uniform3fv( programInfo.uniformLocations.uniformMaterialSpec, specularColor );
  gl.uniform1f( programInfo.uniformLocations.uniformMaterialShini, shininess );
  
  gl.uniform1f( programInfo.uniformLocations.uniformAlphaChannel, 1.0 );
  gl.uniform1i( programInfo.uniformLocations.uniformUsarAtributoColor, 1 );


  gl.uniformMatrix4fv( programInfo.uniformLocations.projectionMatrix, false, projectionMatrix); 
  gl.uniformMatrix4fv( programInfo.uniformLocations.viewMatrix, false, viewMatrix);
  
  gl.uniform3fv( programInfo.uniformLocations.uniformPosition, eye );
  gl.uniform3fv( programInfo.uniformLocations.uniformLightPos, luzpos );

  gl.uniform1i(programInfo.uniformLocations.uniformImageLocation1, 0);  // texture unit 0
  gl.uniform1i(programInfo.uniformLocations.uniformImageLocation2, 1);  // texture unit 1

}

function toRgb(hue, saturation, value){
    let d = 0.0166666666666666 * hue;
	let c = value * saturation;
	let x = c - c * Math.abs(d % 2.0 - 1.0);
	let m = value - c;
	c += m;
	x += m;
    switch (d >>> 0) {
        case 0: return {red: c, green: x, blue: m};
        case 1: return {red: x, green: c, blue: m};
        case 2: return {red: m, green: c, blue: x};
        case 3: return {red: m, green: x, blue: c};
        case 4: return {red: x, green: m, blue: c};
    }
    return {red: c, green: m, blue: x};
};

function drawScene(gl, programInfo, vao, deltaTime, currentTime) {
	
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clearDepth(1.0);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);

  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);	
  gl.uniform1f( programInfo.uniformLocations.uniformTime, currentTime );
  
  
  gl.bindVertexArray(vao);

  // Tell WebGL we want to affect texture unit 0
  gl.activeTexture(gl.TEXTURE0);

  // Bind the texture to texture unit 0
  gl.bindTexture(gl.TEXTURE_2D, textura);
  

  const type = gl.UNSIGNED_SHORT;
  const offset = 0;



  

  CalcularNuevasPosVel();
  //DetectarColisiones();  
  
   /*
  mat4.identity(modelMatrix);
  mat4.translate(modelMatrix, modelMatrix, cuerpo1);//cuerpo1 en lugar de translation  
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);
  
  mat4.identity(modelMatrix);
  mat4.translate(modelMatrix, modelMatrix, cuerpo2);  
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);
  
  mat4.identity(modelMatrix);
  mat4.translate(modelMatrix, modelMatrix, cuerpo3);  
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);
  */
  
  
  mat4.identity(modelMatrix);  
  mat4.translate(modelMatrix, modelMatrix, cuerpo1);
  mat4.scale(modelMatrix, modelMatrix, [1.5,1.5,1.5]);
  mat4.rotate(modelMatrix, modelMatrix, cubeRotation * .8, [0, 1, 0]);
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);  
  
  mat4.identity(modelMatrix);  
  mat4.translate(modelMatrix, modelMatrix, cuerpo2);
  mat4.scale(modelMatrix, modelMatrix, [1.5,1.5,1.5]);
  mat4.rotate(modelMatrix, modelMatrix, cubeRotation * .8, [0, 1, 0]);
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);

  mat4.identity(modelMatrix);
  mat4.translate(modelMatrix, modelMatrix, cuerpo3);
  mat4.scale(modelMatrix, modelMatrix, [1.5,1.5,1.5]);
  mat4.rotate(modelMatrix, modelMatrix, cubeRotation * .8, [0, 1, 0]);
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);  
  
  
  gl.bindVertexArray(null); 

  cubeRotation += deltaTime;

if (needCapture) {
    needCapture = false;
    canvas.toBlob((blob) => {
      saveBlob(blob, `screencapture-${canvas.width}x${canvas.height}.png`);
    });
  }

}//fin de función drawScene();


/*
const elem = document.querySelector('#screenshot');
elem.addEventListener('click', () => {
   needCapture = true;
});

// o también:
  elem.addEventListener('click', () => {
    canvas.toBlob((blob) => {
      saveBlob(blob, `screencapture-${canvas.width}x${canvas.height}.png`);
    });
  });
*/


//función saveBlob para guardar captura de canvas
const saveBlob = (function() {
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    return function saveData(blob, fileName) {
       const url = window.URL.createObjectURL(blob);
       a.href = url;
       a.download = fileName;
       a.click();
    };
  }());

function setCapture(valor) {
            needCapture = valor;
            console.log('Variable needCapture en iFrame cambiada a:', needCapture);
        }




function initShaderProgram(vsSource, fsSource) {
  const vertexShader = loadShader(gl.VERTEX_SHADER, vsSource);
  const fragmentShader = loadShader(gl.FRAGMENT_SHADER, fsSource);
  const shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);

  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    alert('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
    return null;
  }
  return shaderProgram;
}


function loadShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}



function CalcularAceleraciones3D_masa(){
	

      var r12, r13, r23;
	  var m1 = 1.0, m2 = 1.0, m3 = 1.0;

      r12 = Math.sqrt( ( u[6 ]-u[0])*(u[6 ]-u[0] ) + ( u[7 ]-u[1])*(u[7]-u[1 ] ) + ( u[8 ]-u[2])*(u[8 ]-u[2] )  );
      r13 = Math.sqrt( ( u[12]-u[0])*(u[12]-u[0] ) + ( u[13]-u[1])*(u[13]-u[1] ) + ( u[14]-u[2])*(u[14]-u[2] )  );
      r23 = Math.sqrt( ( u[12]-u[6])*(u[12]-u[6] ) + ( u[13]-u[7])*(u[13]-u[7] ) + ( u[14]-u[8])*(u[14]-u[8] )  );

      r12 = r12*r12*r12;
      r13 = r13*r13*r13;
      r23 = r23*r23*r23;

      du[3]  = m2*(  u[6] - u[0]  )/r12 + m3*( u[12] - u[0] )/r13 ;
      du[4]  = m2*(  u[7] - u[1]  )/r12 + m3*( u[13] - u[1] )/r13 ;
      du[5]  = m2*(  u[8] - u[2]  )/r12 + m3*( u[14] - u[2] )/r13 ;

      du[9 ] = m1*(  u[0] - u[6]  )/r12 + m3*( u[12] - u[6] )/r23 ;
      du[10] = m1*(  u[1] - u[7]  )/r12 + m3*( u[13] - u[7] )/r23 ;
      du[11] = m1*(  u[2] - u[8]  )/r12 + m3*( u[14] - u[8] )/r23 ;

      du[15] = m1*(  u[0] - u[12] )/r13 + m2*( u[6] - u[12] )/r23 ;
      du[16] = m1*(  u[1] - u[13] )/r13 + m2*( u[7] - u[13] )/r23 ;
      du[17] = m1*(  u[2] - u[14] )/r13 + m2*( u[8] - u[14] )/r23 ;

}

function derivada3D() {

    CalcularAceleraciones3D_masa();

    for (let iBody = 0; iBody < 3; iBody++) {

        var bodyStart = iBody * 6;

        du[bodyStart + 0] = u[bodyStart + 3];
        du[bodyStart + 1] = u[bodyStart + 4];
        du[bodyStart + 2] = u[bodyStart + 5];

        }
}

function rungeKutta3D(){


	var u0 = new Array();
	var ut = new Array();

    for (let i = 0; i < 18; i++){
        u0.push( u[i] );
        ut.push( 0.0 );
        }


    for( let j = 0; j < 4; j++ )
        {

        derivada3D();

        for(let i = 0; i < 18; i++)
            {
                 u[i] = u0[i] + a[j]*du[i];
                ut[i] = ut[i] + b[j]*du[i];
            }

        }

    for (let i = 0; i < 18; i++) {
        u[i] = u0[i] + ut[i];
      }

}

function CalcularNuevasPosVel() {
	
	//for( let i ; i < 10 ; i++)
		rungeKutta3D();
	

	cuerpo1.length = 0;
	cuerpo2.length = 0;
	cuerpo3.length = 0;	
	cuerpo1.push(u[0],  u[1],  u[2] );
	cuerpo2.push(u[6],  u[7],  u[8] );
	cuerpo3.push(u[12], u[13], u[14]);
	
	

	/*	ambos métodos funcionan el anterior y este:
	cuerpo1[0]=u[0];   cuerpo1[1]=u[1];   cuerpo1[2]=u[2];
	cuerpo2[0]=u[6];   cuerpo2[1]=u[7];   cuerpo2[2]=u[8];
	cuerpo3[0]=u[12];  cuerpo3[1]=u[13];  cuerpo3[2]=u[14];
	*/
	
	for(let i=0;i<3;i++){
		var factor = 10.0;
		cuerpo1[i] = factor*cuerpo1[i];
		cuerpo2[i] = factor*cuerpo2[i];
		cuerpo3[i] = factor*cuerpo3[i];
		
		}
	
	//console.log(cuerpo1[0]);
}

function rebote(incidente, normal){
	//dot(a, b)
	//scale(out, a, b)
	//subtract(out, a, b)	
	//R = 2(N.L)N - L
	
	var refleccion = vec3.fromValues(0, 0, 0);
	var izquierda = vec3.fromValues(0, 0, 0);
	
	var escalar = 2.0*vec3.dot(normal, incidente);
	
	vec3.scale(izquierda, normal ,escalar);

	vec3.subtract(refleccion, izquierda, incidente);
	
	return refleccion;	
}

function DetectarColisiones(){
	  r12 = Math.sqrt( ( u[6 ]-u[0])*(u[6 ]-u[0] ) + ( u[7 ]-u[1])*(u[7]-u[1 ] ) + ( u[8 ]-u[2])*(u[8 ]-u[2] )  );
      r13 = Math.sqrt( ( u[12]-u[0])*(u[12]-u[0] ) + ( u[13]-u[1])*(u[13]-u[1] ) + ( u[14]-u[2])*(u[14]-u[2] )  );
      r23 = Math.sqrt( ( u[12]-u[6])*(u[12]-u[6] ) + ( u[13]-u[7])*(u[13]-u[7] ) + ( u[14]-u[8])*(u[14]-u[8] )  );
	  
	  if(r12<0.2){
			var normal12 = vec3.fromValues(u[6]-u[0], u[7]-u[1], u[8]-u[2]);
			vec3.normalize(normal12, normal12);
			var velocidad1 = vec3.fromValues(u[3], u[4], u[5]);
			
			var velocidad = vec3.fromValues(0, 0, 0);
			
			velocidad = rebote(velocidad1,normal12);
			u[3] = velocidad[0];
			u[4] = velocidad[1];
			u[5] = velocidad[2];
			
			var normal21 = vec3.fromValues(0, 0, 0);
			vec3.negate(normal21, normal12);
			var velocidad2 = vec3.fromValues(u[9], u[10], u[11]);
			
			velocidad = rebote(velocidad2,normal21);
			u[9] = velocidad[0];
			u[10] = velocidad[1];
			u[11] = velocidad[2];
			
			}
	  if(r13<0.2){
			var normal13 = vec3.fromValues(u[12]-u[0], u[13]-u[1], u[14]-u[2]);
			vec3.normalize(normal13, normal13);
			var velocidad1 = vec3.fromValues(u[3], u[4], u[5]);
			
			var velocidad = vec3.fromValues(0, 0, 0);
			
			velocidad = rebote(velocidad1,normal12);
			u[3] = velocidad[0];
			u[4] = velocidad[1];
			u[5] = velocidad[2];
			
			var normal31 = vec3.fromValues(0, 0, 0);
			vec3.negate(normal31, normal13);
			var velocidad2 = vec3.fromValues(u[15], u[16], u[17]);
			
			velocidad = rebote(velocidad2,normal31);
			u[15] = velocidad[0];
			u[16] = velocidad[1];
			u[17] = velocidad[2];
			
			}
			
	  if(r23<0.2){
			var normal23 = vec3.fromValues(u[12]-u[6], u[13]-u[7], u[14]-u[8]);
			vec3.normalize(normal23, normal23);
			var velocidad1 = vec3.fromValues(u[9], u[10], u[11]);
			
			var velocidad = vec3.fromValues(0, 0, 0);
			
			velocidad = rebote(velocidad1,normal23);
			u[9] = velocidad[0];
			u[10] = velocidad[1];
			u[11] = velocidad[2];
			
			var normal32 = vec3.fromValues(0, 0, 0);
			vec3.negate(normal32, normal23);
			var velocidad2 = vec3.fromValues(u[15], u[16], u[17]);
			
			velocidad = rebote(velocidad2,normal32);
			u[15] = velocidad[0];
			u[16] = velocidad[1];
			u[17] = velocidad[2];
			
			}
	
	
}


function resize_canvas(){
	
	canvas.width =window.innerWidth;
	canvas.height=window.innerHeight;
	gl.viewport(0, 0, canvas.width, canvas.height);
	
}


// Initialize a texture and load an image.
// When the image finished loading copy it into the texture.
//
function loadTexture(gl, url) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  // Because images have to be downloaded over the internet
  // they might take a moment until they are ready.
  // Until then put a single pixel in the texture so we can
  // use it immediately. When the image has finished downloading
  // we'll update the texture with the contents of the image.
  const level = 0;
  const internalFormat = gl.RGBA;
  const width = 1;
  const height = 1;
  const border = 0;
  const srcFormat = gl.RGBA;
  const srcType = gl.UNSIGNED_BYTE;
  const pixel = new Uint8Array([0, 0, 255, 255]); // opaque blue
  gl.texImage2D(
    gl.TEXTURE_2D,
    level,
    internalFormat,
    width,
    height,
    border,
    srcFormat,
    srcType,
    pixel
  );

  const image = new Image();
  image.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      level,
      internalFormat,
      srcFormat,
      srcType,
      image
    );

    // WebGL1 has different requirements for power of 2 images
    // vs non power of 2 images so check if the image is a
    // power of 2 in both dimensions.
    if (isPowerOf2(image.width) && isPowerOf2(image.height)) {
      // Yes, it's a power of 2. Generate mips.
      gl.generateMipmap(gl.TEXTURE_2D);
    } else {
      // No, it's not a power of 2. Turn off mips and set
      // wrapping to clamp to edge
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
  };
  image.src = url;

  return texture;
}

function isPowerOf2(value) {
  return (value & (value - 1)) === 0;
}
