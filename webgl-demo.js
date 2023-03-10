var cubeRotation = 0.0;
var modelMatrix      = mat4.create();
var viewMatrix       = mat4.create();
var projectionMatrix = mat4.create();
const fieldOfView = 45 * Math.PI / 180;
const zNear = 0.1;
const zFar = 1000.0;

var eye    = vec3.fromValues(0.0, 0.0, 15.0);
var center = vec3.fromValues(0, 0, 0);
var up     = vec3.fromValues(0, 1, 0);
var luzpos = vec3.fromValues(0.0, 10.0, 0.0);

var cant_puntos_esfera;

let canvas = document.createElement('canvas');
canvas.width =window.innerWidth;
canvas.height=window.innerHeight;
document.body.appendChild(canvas);

//const canvas = document.querySelector('#glcanvas');
const gl = canvas.getContext('webgl2');


var cuerpo1 = new Array();
var cuerpo2 = new Array();
var cuerpo3 = new Array();

var u = [ 0.97000436, -0.24308753, 0.0, -0.466203685 , -0.43236573, 0.0, -0.97000436,  0.24308753, 0.0, -0.466203685 , -0.43236573, 0.0, 0.0, 0.0, 0.0,  0.93240737  ,  0.86473146, 0.0  ];
var du= [  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  ];

const h = 0.01;
const a = [ h/2.0, h/2.0, h,   0.0   ];
const b = [ h/6.0, h/3.0, h/3.0, h/6.0 ];


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

vec3 pos_nueva;
vec3 normal_nueva;
float deltaxz = 0.1;
vec3 dX, dZ;



vec3 mod289(vec3 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x)
{
  return mod289(((x*34.0)+10.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 fade(vec3 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float cnoise(vec3 P)
{
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

// Classic Perlin noise, periodic variant
float pnoise(vec3 P, vec3 rep)
{
  vec3 Pi0 = mod(floor(P), rep); // Integer part, modulo period
  vec3 Pi1 = mod(Pi0 + vec3(1.0), rep); // Integer part + 1, mod period
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}







vec2 pos_horizontal;

void main(){

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
	pos_nueva = aPos + 0.2*cnoise( 4.0*aPos )*aNormal;
	//
	//pos_nueva = pos_nueva + 0.5*cnoise( vec3( length(vec2(pos_nueva.x, pos_nueva.z)),pos_nueva.y, u_time) )*aNormal;
	pos_nueva = pos_nueva + 0.5*cnoise( vec3( pos_nueva.x, pos_nueva.y, u_time) )*aNormal;
	pos_nueva = pos_nueva + 0.2*cnoise( 4.0*pos_nueva )*aNormal + 0.1*cnoise( 8.0*pos_nueva )*aNormal+ 0.05*cnoise( 16.0*pos_nueva )*aNormal;
	

    dX = vec3( aPos.x + deltaxz, u_time, aPos.z );
    dZ = vec3( aPos.x, u_time, aPos.z + deltaxz );
    dX.y = cnoise( dX );
    dZ.y = cnoise( dZ );
    normal_nueva = normalize(cross(dZ - pos_nueva, dX - pos_nueva));





	FragPos = vec3(model * vec4(pos_nueva, 1.0));//pos_nueva
	Normal = mat3(transpose(inverse(model))) * normal_nueva;	//normal_nueva
	 

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

vec3 varColorVertice = vec3(0.0);
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(){


	vec3 ambient, diffuse, specular;


    // ambient
    //vec3 ambient = luz.ambient * material.ambient;


    // diffuse
    vec3 norm = normalize(Normal);
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

        varColorVertice = vec3(fract(0.25*u_time), 0.5 + 0.1*sin(u_time), 1.0);
        //varColorVertice = vec3(fract(0.25*FragPos.y+0.25*u_time), 0.5,  1.0 );
        //varColorVertice = vec3(fract(0.05*length(FragPos)+ u_time), 0.5 , 1.0 );       //fract(sqrt(FragPos.x*FragPos.x+FragPos.z*FragPos.z) );
		varColorVertice = vec3(0.5*sin(u_time)+0.5, 0.5, 1.0);
		//varColorVertice = hsv2rgb( varColorVertice );
		varColorVertice = vec3(0.5*sin(u_time)+0.5, 0.5, 0.0);
		varColorVertice = vec3(ColorVertice);
		
		varColorVertice = vec3(length(FragPos-vec3(model * vec4(0.0,0.0,0.0, 1.0))), 0.5, 0.5);//sqrt(FragPos.x*FragPos.x+FragPos.y*FragPos.y+FragPos.z*FragPos.z)
		varColorVertice = hsv2rgb(varColorVertice);


        ambient  = luz.ambient * varColorVertice;
		 diffuse  = luz.diffuse * diff * varColorVertice;





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
	  uniformCuerPosition: gl.getUniformLocation(shaderProgram, 'cuerPos')
	  
	  
    }
  };
  

  
  const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
  
  mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);
  mat4.lookAt(viewMatrix, eye, center, up);


  const vao = initBuffers(gl, programInfo);
  //const vaoesfera = CrearEsferaVAO(gl, programInfo);

  var then = 0;

  function render(now) {
    now *= 0.001;  // convert to seconds
    const deltaTime = now - then;
    then = now;

    drawScene(gl, programInfo, vao, deltaTime, now);

    requestAnimationFrame(render);
  }
  
  requestAnimationFrame(render);
  }

function CrearEsferaVAO(programInfo){

    var num_lon = 14, num_lat = 7, num_puntos;
    var lat, lon, x, y, z;

    num_puntos = num_lat*num_lon + 2;//sumar 2 por los polos
    cant_puntos_esfera = 2*3*num_lon*num_lat;


    var vertices = [];	//	3*num_puntos
    var normales = [];	//	3*num_puntos
    var  colores = [];	//	3*num_puntos
    var  indices = [];	//	2*3*(num_lon)*(num_lat)
	var   puntos = [];

	let polonorte = {
		"x": 0.0,
		"y": 0.0,
		"z": 1.0
	};

	let polosur = {
		"x": 0.0,
		"y": 0.0,
		"z": -1.0
	};
	
    puntos.push(polonorte);
	let obj_punto = {};
    for ( let paral = 1; paral <= num_lat; paral++)
        for ( let merid = 0; merid < num_lon; merid++){

            lat = 0.5*Math.PI - Math.PI*paral/num_lat+1;
            lon = 2.0*Math.PI*merid/num_lon;


            x = Math.cos(lon)*Math.cos(lat);
            y = Math.sin(lon)*Math.cos(lat);
            z = Math.sin(lat);
			
			obj_punto.x = x;  // = {"x" : x, "y" : y, "z" : z };
			obj_punto.y = y;
			obj_punto.z = z;

            puntos.push(obj_punto);
    }
    puntos.push(polosur);


    var n = 0;
    for( let i = 0 ; i < num_puntos ; i++){

            vertices[n]   = puntos[i].x;
            vertices[n+1] = puntos[i].z;
            vertices[n+2] = puntos[i].y;

            colores[n]   = 1.0;
            colores[n+1] = 1.0;
            colores[n+2] = 1.0;

            normales[n]   = puntos[i].x;
            normales[n+1] = puntos[i].z;
            normales[n+2] = puntos[i].y;

            n += 3;
        }


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
	
	var vao = CrearVAO(gl, programInfo, vertices, normales, colores, indices);
	return vao;
};

function CrearVAO(programInfo, vertices, normales, colores, indices ){
	
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
  


  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
	  


  gl.bindVertexArray(null);
  
  return vao;	
	
};



function initBuffers(gl, programInfo) {

    var num_lon = 36, num_lat = 18, num_puntos;
    var lat, lon, x, y, z;

    num_puntos = num_lat*num_lon + 2;//sumar 2 por los polos
    cant_puntos_esfera = 2*3*num_lon*num_lat;


    var vertices = new Array();	//	3*num_puntos
    var normales = new Array();	//	3*num_puntos
    var  colores = new Array();	//	3*num_puntos
    var  indices = new Array();	//	2*3*(num_lon)*(num_lat) //en webgl existe un límite de 65k para los valores de los indices no así para la cantidad de indices 



    vertices.push(0.0, 0.0, 1.0);
	normales.push(0.0, 0.0, 1.0);
	colores.push(Math.random(), Math.random(), Math.random());

    for ( let paral = 1; paral <= num_lat; paral++)
        for ( let merid = 0; merid < num_lon; merid++){

            lat = 0.5*Math.PI - (Math.PI*paral)/(num_lat+1);
            lon = (2.0*Math.PI*merid)/num_lon;


            x = Math.cos(lon)*Math.cos(lat);
            y = Math.sin(lon)*Math.cos(lat);
            z = Math.sin(lat);

            vertices.push(x, y, z);
			normales.push(x, y, z);
			var rgb = toRgb(360*(0.5*x+0.5), 0.5, 0.5);
			colores.push(rgb.red, rgb.green, rgb.blue );
			
			//colores.push(Math.random(), Math.random(), Math.random());
    }
    vertices.push(0.0, 0.0, -1.0);
	normales.push(0.0, 0.0, -1.0);
	colores.push(Math.random(), Math.random(), Math.random());


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
    var  _indices = new Uint16Array(2*3*(num_lon)*(num_lat));	//	2*3*(num_lon)*(num_lat)
	
	for(let i = 0 ; i < vertices.length; i++){
		_vertices[i] = vertices[i];
		_normales[i] = normales[i];
		_colores[i]  = colores[i];
		
		}
	for(let i = 0 ; i < indices.length; i++){
		_indices[i] = indices[i];
		
		}

	console.log( gl.getParameter(gl.MAX_ELEMENT_INDEX) + " 1 " );
	console.log( gl.getParameter(gl.MAX_ELEMENTS_VERTICES) + " 2" );
	console.log( gl.getParameter(gl.MAX_ELEMENTS_INDICES) + " 3" );
	console.log( gl.getParameter(gl.MAX_ELEMENT_INDEX) + " 4" );
	
  var vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  var numComponents = 3;
  var type = gl.FLOAT;
  var normalize = false;
  var stride = 0;
  var offset = 0;

  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(_vertices), gl.STATIC_DRAW);
  gl.enableVertexAttribArray( programInfo.attribLocations.vertexPosition);
  gl.vertexAttribPointer( programInfo.attribLocations.vertexPosition, numComponents, type, normalize, stride, offset);
  

  const colorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(_colores), gl.STATIC_DRAW);
  gl.enableVertexAttribArray( programInfo.attribLocations.vertexColor);
  gl.vertexAttribPointer( programInfo.attribLocations.vertexColor, numComponents, type, normalize, stride, offset);
  
  
  const normalBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(_normales), gl.STATIC_DRAW);
  gl.enableVertexAttribArray( programInfo.attribLocations.vertexNormal);
  gl.vertexAttribPointer( programInfo.attribLocations.vertexNormal, numComponents, type, normalize, stride, offset);
  


  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(_indices), gl.STATIC_DRAW);
	  


  gl.bindVertexArray(null);
  
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


  return vao; 

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

 
  
  //mat4.identity(modelMatrix);
  //mat4.translate(modelMatrix, modelMatrix, [-0.0, 0.0, -6.0]);
  //mat4.rotate(modelMatrix, modelMatrix, cubeRotation,      [0, 0, 1]);
  //mat4.rotate(modelMatrix, modelMatrix, cubeRotation * .7, [0, 1, 0]);
  //mat4.rotate(modelMatrix, modelMatrix, cubeRotation * .3, [1, 0, 0]);
  //mat4.scale(modelMatrix, modelMatrix, [0.2*Math.cos(currentTime+1.0)+0.5, 0.2*Math.sin(currentTime)+0.5, 1.0]);
  
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);	
  gl.uniform1f( programInfo.uniformLocations.uniformTime, currentTime );
  
  
  gl.bindVertexArray(vao);
  
  CalcularNuevasPosVel();
  
  {
    const vertexCount = 60;//36
    const type = gl.UNSIGNED_SHORT;
    const offset = 0;
      /*gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);*/
  
  

  

  
  //var translation = vec3.create();

  //uniformCuerPosition
  
  mat4.identity(modelMatrix);
  //vec3.set (translation, cuerpo1[0], cuerpo1[1], cuerpo1[2]);
  //mat4.rotate(modelMatrix, modelMatrix, cubeRotation * .7, [0, 1, 0]);
  mat4.translate(modelMatrix, modelMatrix, cuerpo1);//cuerpo1 en lugar de translation  
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);
  
  mat4.identity(modelMatrix);
  //vec3.set (translation, cuerpo2[0], cuerpo2[1], cuerpo2[2]);
  //mat4.rotate(modelMatrix, modelMatrix, cubeRotation * .7, [1, 0, 0]);
  mat4.translate(modelMatrix, modelMatrix, cuerpo2);  
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);
  
  mat4.identity(modelMatrix);
  //vec3.set (translation, cuerpo3[0], cuerpo3[1], cuerpo3[2]);
  //mat4.rotate(modelMatrix, modelMatrix, cubeRotation * .7, [0, 0, 1]);
  mat4.translate(modelMatrix, modelMatrix, cuerpo3);  
  gl.uniformMatrix4fv( programInfo.uniformLocations.modelMatrix, false, modelMatrix);
  gl.drawElements(gl.TRIANGLES, cant_puntos_esfera, type, offset);
  
  gl.bindVertexArray(null);
  
  }
  
  
  

  cubeRotation += deltaTime;
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

function my_reflect(incidente, normal){
	//dot(a, b)
	//scale(out, a, b)
	//subtract(out, a, b)	
	//R = 2(N.L)N - L
	
	var reflection = vec3.fromValues(0, 0, 0);
	var izquierda = vec3.fromValues(0, 0, 0);
	
	var escalar = 2.0*dot(normal, incidente);
	
	scale(izquierda, normal ,escalar);

	subtract(reflection, izquierda, incidente);
	
	return reflection;	
}


function resize_canvas(){
	
	canvas.width =window.innerWidth;
	canvas.height=window.innerHeight;
	
}